//! Guardrail provider-registry and engine factory (task 13.1).
//!
//! Builds a [`ProviderRegistry`] from a [`GuardrailConfig`] by mapping each
//! [`GuardrailProviderConfig`] to its concrete `Arc<dyn GuardrailProvider>`
//! backend, then assembles a [`GuardrailEngine`] from the registry and the
//! configured pipelines. This is the single place where declared provider
//! settings are translated into live provider instances shared across requests
//! (Req 8.5), and the construction path used both at startup and on hot-reload
//! (Req 1.8; wiring lives in `gateway/mod.rs`).

use std::sync::Arc;

use reqwest::Client;

use crate::cache::SemanticCache;
use crate::guardrail::config::{GuardrailConfig, GuardrailProviderConfig, GuardrailProviderType};
use crate::guardrail::pipeline::PipelineResolverError;
use crate::guardrail::provider::{GuardrailProvider, ProviderRegistry};
use crate::guardrail::providers::custom_http::CustomHttpProvider;
use crate::guardrail::providers::lakera::LakeraProvider;
use crate::guardrail::providers::moderation::OpenAiModerationProvider;
use crate::guardrail::providers::presidio::PresidioProvider;
use crate::guardrail::providers::regex::{RegexCompileError, RegexProvider};
use crate::guardrail::providers::semantic::SemanticProvider;
use crate::guardrail::providers::unicode_stego::UnicodeStegoProvider;
use crate::guardrail::GuardrailEngine;
use crate::metrics::Metrics;
use crate::security::outbound_fetch::{self, OutboundFetchConfig};

/// Errors produced while translating a [`GuardrailConfig`] into a live
/// [`ProviderRegistry`] / [`GuardrailEngine`].
///
/// Configuration validation (`config/validation.rs`) is expected to have
/// rejected most invalid shapes already; a value surfacing here is treated as a
/// startup/hot-reload failure by the caller.
#[derive(Debug)]
pub enum RegistryBuildError {
    /// A regex provider's patterns failed to compile.
    RegexCompile {
        provider: String,
        source: RegexCompileError,
    },
    /// A required setting for a provider type was absent.
    MissingSetting {
        provider: String,
        field: &'static str,
    },
    /// A `semantic` provider was declared but no semantic cache is configured,
    /// so its embedding provider/model and Qdrant instance cannot be reused
    /// (Req 7.1, 7.5).
    SemanticCacheUnavailable { provider: String },
    /// Assembling the pipeline resolver from the registry failed (a stage
    /// referenced an unknown provider). Should be unreachable after validation.
    Resolver(PipelineResolverError),
    /// An HTTP-backed provider's configured endpoint was rejected by the
    /// outbound-fetch (SSRF) guard — forbidden address class, scheme, or
    /// unresolvable host. Fail closed: no client is built for that target
    /// (Req 2.1).
    OutboundFetchRejected {
        provider: String,
        host: String,
        reason: String,
    },
    /// Building the redirect-guarded HTTP client for a custom_http provider
    /// failed at the reqwest layer.
    ClientBuild { provider: String, source: String },
}

impl std::fmt::Display for RegistryBuildError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegistryBuildError::RegexCompile { provider, source } => write!(
                f,
                "guardrail provider '{provider}' has an invalid regex pattern: {source}"
            ),
            RegistryBuildError::MissingSetting { provider, field } => write!(
                f,
                "guardrail provider '{provider}' is missing required setting '{field}'"
            ),
            RegistryBuildError::SemanticCacheUnavailable { provider } => write!(
                f,
                "guardrail provider '{provider}' requires a configured semantic cache \
                 (embedding provider + Qdrant), but none is available"
            ),
            RegistryBuildError::Resolver(e) => {
                write!(f, "guardrail pipeline resolution failed: {e}")
            }
            RegistryBuildError::OutboundFetchRejected {
                provider,
                host,
                reason,
            } => write!(
                f,
                "guardrail provider '{provider}' endpoint host {host:?} rejected by \
outbound-fetch guard: {reason}"
            ),
            RegistryBuildError::ClientBuild { provider, source } => write!(
                f,
                "guardrail provider '{provider}' failed to build its HTTP client: {source}"
            ),
        }
    }
}

impl std::error::Error for RegistryBuildError {}

impl From<PipelineResolverError> for RegistryBuildError {
    fn from(e: PipelineResolverError) -> Self {
        RegistryBuildError::Resolver(e)
    }
}

/// Build a [`ProviderRegistry`] from `config`, instantiating one shared
/// `Arc<dyn GuardrailProvider>` per declared provider (Req 8.5).
///
/// `http_client` is reused by every HTTP-backed provider (presidio,
/// custom_http, openai_moderation, lakera) — except custom_http, whose
/// operator-supplied endpoint is additionally guarded by the outbound-fetch
/// (SSRF) validator and gets a redirect-guarded client of its own
/// (Req 2.1). `semantic_cache` supplies the embedding provider/model and
/// Qdrant instance reused by any `semantic` provider (Req 7.1, 7.5); when
/// `None`, declaring a `semantic` provider is a build error.
pub fn build_registry(
    config: &GuardrailConfig,
    http_client: &Client,
    semantic_cache: Option<&SemanticCache>,
    outbound: &OutboundFetchConfig,
) -> Result<ProviderRegistry, RegistryBuildError> {
    let mut registry = ProviderRegistry::new();

    for provider_cfg in &config.providers {
        let provider = build_provider(provider_cfg, http_client, semantic_cache, outbound)?;
        registry.insert(provider_cfg.name.clone(), provider);
    }

    Ok(registry)
}

/// Build the full [`GuardrailEngine`]: registry + pre-compiled pipelines.
///
/// This is the construction path used by `GatewayServer::new` at startup and by
/// `apply_runtime_config_update` on hot-reload (Req 1.8).
pub fn build_engine(
    config: &GuardrailConfig,
    http_client: &Client,
    semantic_cache: Option<&SemanticCache>,
    metrics: Option<Arc<Metrics>>,
    outbound: &OutboundFetchConfig,
) -> Result<GuardrailEngine, RegistryBuildError> {
    let registry = build_registry(config, http_client, semantic_cache, outbound)?;
    let max_entries = config.max_reinjection_entries;
    let engine = GuardrailEngine::new_with_capacity(config, &registry, metrics, max_entries)?;
    Ok(engine)
}

/// Instantiate a single provider from its config entry.
fn build_provider(
    cfg: &GuardrailProviderConfig,
    http_client: &Client,
    semantic_cache: Option<&SemanticCache>,
    outbound: &OutboundFetchConfig,
) -> Result<Arc<dyn GuardrailProvider>, RegistryBuildError> {
    let settings = &cfg.settings;
    let timeout = Some(cfg.timeout_seconds);

    let provider: Arc<dyn GuardrailProvider> = match cfg.provider_type {
        GuardrailProviderType::Regex => {
            let regex = RegexProvider::new(&settings.patterns).map_err(|source| {
                RegistryBuildError::RegexCompile {
                    provider: cfg.name.clone(),
                    source,
                }
            })?;
            Arc::new(regex)
        }
        GuardrailProviderType::Presidio => {
            let endpoint = settings
                .endpoint
                .clone()
                .ok_or(RegistryBuildError::MissingSetting {
                    provider: cfg.name.clone(),
                    field: "endpoint",
                })?;
            Arc::new(PresidioProvider::new(
                http_client.clone(),
                endpoint,
                settings.entities.clone(),
                settings.language.clone(),
                settings.confidence_threshold,
                timeout,
            ))
        }
        GuardrailProviderType::CustomHttp => {
            // `url` is the documented setting; accept `endpoint` as an alias.
            let url = settings
                .url
                .clone()
                .or_else(|| settings.endpoint.clone())
                .ok_or(RegistryBuildError::MissingSetting {
                    provider: cfg.name.clone(),
                    field: "url",
                })?;
            // Caller-tier outbound-fetch guard (Req 2.1): validate the
            // operator-supplied endpoint before any client is built. Rejection
            // carries host + reason only (never the credential-bearing URL) and
            // fails closed — no client is constructed for a forbidden target.
            let parsed = reqwest::Url::parse(&url).map_err(|_| {
                let host = url
                    .split("://")
                    .nth(1)
                    .and_then(|rest| rest.split(['/', '?', '#']).next())
                    .unwrap_or_default()
                    .to_owned();
                RegistryBuildError::OutboundFetchRejected {
                    provider: cfg.name.clone(),
                    host,
                    reason: "endpoint is not a valid URL".to_owned(),
                }
            })?;
            if let Err(error) = outbound_fetch::validate_outbound_url(
                &parsed,
                outbound,
                &outbound_fetch::std_resolve_host,
            ) {
                let (host, reason) = error.host_and_reason();
                tracing::warn!(
                target: "security::outbound_fetch",
                provider = %cfg.name,
                host = %host,
                reason = %reason,
                "guardrail custom_http endpoint rejected by outbound-fetch guard"
                );
                return Err(RegistryBuildError::OutboundFetchRejected {
                    provider: cfg.name.clone(),
                    host: host.to_owned(),
                    reason,
                });
            }
            // Redirect-guarded client: every hop re-resolves and re-validates
            // against the same rules (mirrors admin/mod.rs wiring). The per-provider
            // `timeout_seconds` keeps applying at the request layer.
            let guarded_client = Client::builder()
                .redirect(outbound_fetch::guarded_redirect_policy(
                    outbound.clone(),
                    Arc::new(outbound_fetch::std_resolve_host),
                ))
                .build()
                .map_err(|source| RegistryBuildError::ClientBuild {
                    provider: cfg.name.clone(),
                    source: source.to_string(),
                })?;
            Arc::new(CustomHttpProvider::new(guarded_client, url, timeout))
        }
        GuardrailProviderType::OpenaiModeration => {
            let api_key_env =
                settings
                    .api_key_env
                    .as_deref()
                    .ok_or(RegistryBuildError::MissingSetting {
                        provider: cfg.name.clone(),
                        field: "api_key_env",
                    })?;
            Arc::new(OpenAiModerationProvider::new(
                http_client.clone(),
                settings.endpoint.clone(),
                None,
                api_key_env,
                timeout,
            ))
        }
        GuardrailProviderType::Lakera => {
            let api_key_env =
                settings
                    .api_key_env
                    .as_deref()
                    .ok_or(RegistryBuildError::MissingSetting {
                        provider: cfg.name.clone(),
                        field: "api_key_env",
                    })?;
            Arc::new(LakeraProvider::new(
                http_client.clone(),
                settings.endpoint.clone(),
                api_key_env,
                timeout,
            ))
        }
        GuardrailProviderType::Semantic => {
            let cache = semantic_cache.ok_or(RegistryBuildError::SemanticCacheUnavailable {
                provider: cfg.name.clone(),
            })?;
            let allow_collection =
                settings
                    .allow_collection
                    .clone()
                    .ok_or(RegistryBuildError::MissingSetting {
                        provider: cfg.name.clone(),
                        field: "allow_collection",
                    })?;
            let deny_collection =
                settings
                    .deny_collection
                    .clone()
                    .ok_or(RegistryBuildError::MissingSetting {
                        provider: cfg.name.clone(),
                        field: "deny_collection",
                    })?;
            Arc::new(SemanticProvider::from_semantic_cache(
                cache,
                allow_collection,
                deny_collection,
                settings.allow_threshold,
                settings.deny_threshold,
            ))
        }
        // Local, deterministic: no HTTP client, no external settings required.
        GuardrailProviderType::UnicodeStego => {
            Arc::new(UnicodeStegoProvider::new(&settings.unicode_stego))
        }
    };

    Ok(provider)
}
#[cfg(test)]
mod tests {
    use super::*;
    use crate::guardrail::config::{
        FailurePolicy, GuardrailProviderConfig, GuardrailProviderType, InstructionInsertionMode,
        PipelineConfig, PolicyAction, ProviderSettings, RegexPatternConfig, RegexRuleMode,
        StageConfig, StagePhase,
    };

    /// Strict outbound-fetch defaults (no private hosts, https only) — the
    /// production default posture.
    fn strict_outbound() -> OutboundFetchConfig {
        OutboundFetchConfig::default()
    }

    fn regex_provider(name: &str) -> GuardrailProviderConfig {
        GuardrailProviderConfig {
            name: name.to_string(),
            provider_type: GuardrailProviderType::Regex,
            failure_policy: FailurePolicy::FailOpen,
            timeout_seconds: 5,
            settings: ProviderSettings {
                patterns: vec![RegexPatternConfig {
                    name: "email".to_string(),
                    regex: r"\w+@\w+\.\w+".to_string(),
                    entity: "EMAIL".to_string(),
                    mode: RegexRuleMode::Deny,
                }],
                ..Default::default()
            },
        }
    }

    fn presidio_provider(name: &str, endpoint: Option<&str>) -> GuardrailProviderConfig {
        GuardrailProviderConfig {
            name: name.to_string(),
            provider_type: GuardrailProviderType::Presidio,
            failure_policy: FailurePolicy::FailOpen,
            timeout_seconds: 5,
            settings: ProviderSettings {
                endpoint: endpoint.map(str::to_string),
                entities: vec!["EMAIL_ADDRESS".to_string()],
                confidence_threshold: Some(0.5),
                ..Default::default()
            },
        }
    }

    #[test]
    fn builds_registry_with_multiple_provider_types() {
        let config = GuardrailConfig {
            providers: vec![
                regex_provider("scanner"),
                presidio_provider("pii", Some("http://presidio:5001/analyze")),
                GuardrailProviderConfig {
                    name: "custom".to_string(),
                    provider_type: GuardrailProviderType::CustomHttp,
                    failure_policy: FailurePolicy::FailClose,
                    timeout_seconds: 3,
                    settings: ProviderSettings {
                        url: Some("http://scanner:8080/scan".to_string()),
                        ..Default::default()
                    },
                },
                GuardrailProviderConfig {
                    name: "mod".to_string(),
                    provider_type: GuardrailProviderType::OpenaiModeration,
                    failure_policy: FailurePolicy::FailOpen,
                    timeout_seconds: 5,
                    settings: ProviderSettings {
                        api_key_env: Some("OPENAI_API_KEY".to_string()),
                        ..Default::default()
                    },
                },
                GuardrailProviderConfig {
                    name: "lakera".to_string(),
                    provider_type: GuardrailProviderType::Lakera,
                    failure_policy: FailurePolicy::FailOpen,
                    timeout_seconds: 5,
                    settings: ProviderSettings {
                        api_key_env: Some("LAKERA_API_KEY".to_string()),
                        ..Default::default()
                    },
                },
            ],
            ..Default::default()
        };

        let client = Client::new();
        let outbound = permissive_outbound();
        let registry = build_registry(&config, &client, None, &outbound).expect("registry builds");

        assert_eq!(registry.len(), 5);
        assert_eq!(registry.get("scanner").unwrap().provider_type(), "regex");
        assert_eq!(registry.get("pii").unwrap().provider_type(), "presidio");
        assert_eq!(
            registry.get("custom").unwrap().provider_type(),
            "custom_http"
        );
        assert_eq!(
            registry.get("mod").unwrap().provider_type(),
            "openai_moderation"
        );
        assert_eq!(registry.get("lakera").unwrap().provider_type(), "lakera");
    }

    #[test]
    fn build_engine_compiles_pipelines() {
        let config = GuardrailConfig {
            providers: vec![regex_provider("scanner")],
            pipelines: vec![PipelineConfig {
                name: "default".to_string(),
                stages: vec![StageConfig {
                    name: "block-email".to_string(),
                    provider: "scanner".to_string(),
                    phase: StagePhase::PreCall,
                    action: PolicyAction::Block,
                }],
                redaction_notice_instruction: None,
                instruction_insertion_mode: InstructionInsertionMode::default(),
                failover_on_refusal: false,
                refusal_phrase_list: None,
                tool_result: crate::guardrail::config::ToolResultPhaseConfig::default(),
            }],
            global_default_pipeline: Some("default".to_string()),
            ..Default::default()
        };

        let client = Client::new();
        let engine =
            build_engine(&config, &client, None, None, &strict_outbound()).expect("engine builds");
        // The global-default pipeline resolves to its single compiled stage.
        let stages = engine
            .resolver()
            .resolve(&crate::guardrail::pipeline::BindingSelector::default());
        assert_eq!(stages.len(), 1);
        assert_eq!(stages[0].stage_name, "block-email");
        assert_eq!(stages[0].provider_type, "regex");
    }

    #[test]
    fn regex_compile_failure_is_surfaced() {
        let mut cfg = regex_provider("bad");
        cfg.settings.patterns[0].regex = "(unclosed".to_string();
        let config = GuardrailConfig {
            providers: vec![cfg],
            ..Default::default()
        };

        let client = Client::new();
        let err = build_registry(&config, &client, None, &strict_outbound())
            .expect_err("invalid regex must fail");
        assert!(matches!(err, RegistryBuildError::RegexCompile { .. }));
    }

    #[test]
    fn missing_presidio_endpoint_is_error() {
        let config = GuardrailConfig {
            providers: vec![presidio_provider("pii", None)],
            ..Default::default()
        };
        let client = Client::new();
        let err = build_registry(&config, &client, None, &strict_outbound())
            .expect_err("missing endpoint fails");
        assert!(matches!(
            err,
            RegistryBuildError::MissingSetting {
                field: "endpoint",
                ..
            }
        ));
    }

    #[test]
    fn semantic_without_cache_is_error() {
        let config = GuardrailConfig {
            providers: vec![GuardrailProviderConfig {
                name: "sem".to_string(),
                provider_type: GuardrailProviderType::Semantic,
                failure_policy: FailurePolicy::FailOpen,
                timeout_seconds: 5,
                settings: ProviderSettings {
                    allow_collection: Some("allow".to_string()),
                    deny_collection: Some("deny".to_string()),
                    ..Default::default()
                },
            }],
            ..Default::default()
        };
        let client = Client::new();
        let err = build_registry(&config, &client, None, &strict_outbound())
            .expect_err("semantic needs cache");
        assert!(matches!(
            err,
            RegistryBuildError::SemanticCacheUnavailable { .. }
        ));
    }

    fn custom_http_provider(url: &str) -> GuardrailProviderConfig {
        GuardrailProviderConfig {
            name: "custom".to_string(),
            provider_type: GuardrailProviderType::CustomHttp,
            failure_policy: FailurePolicy::FailClose,
            timeout_seconds: 3,
            settings: ProviderSettings {
                url: Some(url.to_string()),
                ..Default::default()
            },
        }
    }

    fn permissive_outbound() -> OutboundFetchConfig {
        OutboundFetchConfig {
            caller: crate::security::outbound_fetch::CallerFetchConfig {
                allow_private_hosts: true,
                allow_http: true,
                ..Default::default()
            },
        }
    }

    /// Literal link-local (cloud metadata) endpoints must fail closed: the
    /// registry build errors and no client is constructed for the target
    /// (Req 2.1). Literal IPs take no DNS. https scheme isolates the
    /// forbidden-address class as the rejection reason.
    #[test]
    fn custom_http_metadata_endpoint_fails_closed() {
        let config = GuardrailConfig {
            providers: vec![custom_http_provider(
                "https://169.254.169.254/latest/meta-data",
            )],
            ..Default::default()
        };
        let client = Client::new();
        let err = build_registry(&config, &client, None, &strict_outbound())
            .expect_err("metadata endpoint must fail closed");
        match err {
            RegistryBuildError::OutboundFetchRejected {
                provider,
                host,
                reason,
            } => {
                assert_eq!(provider, "custom");
                assert_eq!(host, "169.254.169.254");
                assert!(reason.contains("link-local"), "unexpected reason: {reason}");
            }
            other => panic!("expected OutboundFetchRejected, got {other:?}"),
        }
    }

    /// With `allow_private_hosts: true` the same loopback-class endpoint is
    /// permitted and the provider builds (positive flip case).
    #[test]
    fn custom_http_private_endpoint_allowed_when_configured() {
        let config = GuardrailConfig {
            providers: vec![custom_http_provider(
                "http://169.254.169.254/latest/meta-data",
            )],
            ..Default::default()
        };
        let client = Client::new();
        let registry = build_registry(&config, &client, None, &permissive_outbound())
            .expect("allow_private_hosts permits the endpoint");
        assert_eq!(
            registry.get("custom").unwrap().provider_type(),
            "custom_http"
        );
    }
}
