//! Guardrail provider trait and core analysis types.
//!
//! A [`GuardrailProvider`] is a pluggable content-analysis backend (regex,
//! presidio, openai_moderation, lakera, custom_http, semantic). Providers are
//! instantiated once at configuration load time and shared across concurrent
//! requests as `Arc<dyn GuardrailProvider>` via a [`ProviderRegistry`]
//! (Req 8.5). The pipeline engine calls [`GuardrailProvider::analyze`] on the
//! (already length-clamped) content and enforces the stage's policy action
//! against the returned [`Finding`]s (Req 8.1).

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use thiserror::Error;

use crate::guardrail::config::FailurePolicy;

/// A single detected policy-relevant span within analyzed content.
///
/// Byte offsets are relative to the exact `content` string passed to
/// [`GuardrailProvider::analyze`]; `content[start..end]` is the matched span.
#[derive(Debug, Clone, PartialEq)]
pub struct Finding {
    /// Entity label, at most 128 chars (e.g. `EMAIL_ADDRESS`, `US_SSN`,
    /// `prompt_injection`).
    pub entity_label: String,
    /// Byte-offset start within the analyzed content.
    pub start: usize,
    /// Byte-offset end (exclusive) within the analyzed content.
    pub end: usize,
    /// Optional matched text, populated by regex/presidio providers and used to
    /// determine masking length.
    pub matched_text: Option<String>,
    /// Optional match score (semantic similarity or moderation confidence).
    pub score: Option<f32>,
}

/// Errors returned by a [`GuardrailProvider::analyze`] call.
///
/// The engine maps these (together with a timeout) onto the provider's
/// configured failure policy (fail_open / fail_close) — see task 2.2.
#[derive(Debug, Error)]
pub enum GuardrailProviderError {
    /// The backend service was unreachable or the transport failed.
    #[error("guardrail provider unreachable: {0}")]
    Unreachable(String),

    /// The backend returned a non-success HTTP status.
    #[error("guardrail provider returned status {status}: {message}")]
    UpstreamStatus { status: u16, message: String },

    /// The backend response did not conform to the expected schema.
    #[error("guardrail provider returned malformed response: {0}")]
    MalformedResponse(String),

    /// The provider could not complete analysis for an internal reason.
    #[allow(dead_code)] // part of the provider error API; not constructed in the binary build
    #[error("guardrail provider analysis failed: {0}")]
    Analysis(String),
}

/// A pluggable guardrail content-analysis backend (Req 8.1).
///
/// Implementations must be `Send + Sync` because a single instance is shared
/// across concurrent requests behind an `Arc`. The engine guarantees that
/// `content` is at most 100,000 UTF-8 characters before `analyze` is called.
#[async_trait::async_trait]
pub trait GuardrailProvider: Send + Sync {
    /// Analyze `content` and return the detected findings.
    ///
    /// Returns an empty vector when no policy-relevant content is detected.
    async fn analyze(&self, content: &str) -> Result<Vec<Finding>, GuardrailProviderError>;

    /// Provider type discriminant, used as a metric label (e.g. `"regex"`).
    fn provider_type(&self) -> &'static str;
}

/// Registry mapping declared provider name → shared provider instance (Req 8.5).
///
/// Providers are constructed once at configuration load time and looked up by
/// the name a [`crate::guardrail::config::StageConfig`] references.
#[derive(Clone, Default)]
pub struct ProviderRegistry {
    providers: HashMap<String, Arc<dyn GuardrailProvider>>,
}

impl ProviderRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self {
            providers: HashMap::new(),
        }
    }

    /// Insert a provider under `name`, returning any previously registered
    /// provider with the same name.
    pub fn insert(
        &mut self,
        name: impl Into<String>,
        provider: Arc<dyn GuardrailProvider>,
    ) -> Option<Arc<dyn GuardrailProvider>> {
        self.providers.insert(name.into(), provider)
    }

    /// Look up a shared provider instance by declared name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn GuardrailProvider>> {
        self.providers.get(name).cloned()
    }

    /// Return `true` if a provider is registered under `name`.
    #[allow(dead_code)] // used by tests; unused in the binary build
    pub fn contains(&self, name: &str) -> bool {
        self.providers.contains_key(name)
    }

    /// Number of registered providers.
    #[allow(dead_code)] // used by tests; unused in the binary build
    pub fn len(&self) -> usize {
        self.providers.len()
    }

    /// Return `true` if no providers are registered.
    #[allow(dead_code)] // public API / test-only; unused in the binary build
    pub fn is_empty(&self) -> bool {
        self.providers.is_empty()
    }
}

impl std::fmt::Debug for ProviderRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // `dyn GuardrailProvider` is not `Debug`; list registered names instead.
        f.debug_struct("ProviderRegistry")
            .field("providers", &self.providers.keys().collect::<Vec<_>>())
            .finish()
    }
}

/// Outcome of running a single stage's provider under its failure policy.
///
/// Produced by [`analyze_with_policy`], which wraps [`GuardrailProvider::analyze`]
/// in a timeout and maps errors/timeouts onto the provider's configured
/// [`FailurePolicy`] (Req 8.6, 8.7).
///
/// Superseded in the engine's hot path by its own `StageAnalysis` (which also
/// distinguishes scan timeouts); retained as reusable library API and exercised
/// by the provider integration tests, so it is unused in the binary build.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub enum StageDisposition {
    /// The provider completed successfully; these findings drive the stage's
    /// policy action.
    Findings(Vec<Finding>),
    /// The provider failed or timed out under a `fail_open` policy: skip this
    /// stage and continue the pipeline (Req 9.6).
    SkipOpen,
    /// The provider failed or timed out under a `fail_close` policy: halt the
    /// pipeline and reject the request with a guardrail-unavailable error
    /// (Req 9.7).
    HaltClosed,
}

/// Run a provider's [`analyze`](GuardrailProvider::analyze) under a timeout and
/// resolve errors/timeouts through the stage's [`FailurePolicy`] (Req 8.6, 8.7).
///
/// On success returns [`StageDisposition::Findings`]. On an `Err` from the
/// provider or a timeout after `timeout_duration`, the disposition is derived
/// from `policy`: [`FailurePolicy::FailOpen`] → [`StageDisposition::SkipOpen`],
/// [`FailurePolicy::FailClose`] → [`StageDisposition::HaltClosed`].
///
/// `content` is expected to already be clamped to the engine's UTF-8 limit.
#[allow(dead_code)] // library API exercised by integration tests; unused in the binary build
pub async fn analyze_with_policy(
    provider: &dyn GuardrailProvider,
    content: &str,
    timeout_duration: Duration,
    policy: FailurePolicy,
) -> StageDisposition {
    match tokio::time::timeout(timeout_duration, provider.analyze(content)).await {
        // Provider completed within the budget and succeeded.
        Ok(Ok(findings)) => StageDisposition::Findings(findings),
        // Provider completed within the budget but returned an error.
        Ok(Err(_)) => disposition_for_failure(policy),
        // Provider exceeded the timeout budget.
        Err(_elapsed) => disposition_for_failure(policy),
    }
}

/// Map a provider failure (error or timeout) onto a [`StageDisposition`] per the
/// configured [`FailurePolicy`].
#[allow(dead_code)] // helper for `analyze_with_policy`; unused in the binary build
fn disposition_for_failure(policy: FailurePolicy) -> StageDisposition {
    match policy {
        FailurePolicy::FailOpen => StageDisposition::SkipOpen,
        FailurePolicy::FailClose => StageDisposition::HaltClosed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StubProvider(&'static str);

    #[async_trait::async_trait]
    impl GuardrailProvider for StubProvider {
        async fn analyze(&self, content: &str) -> Result<Vec<Finding>, GuardrailProviderError> {
            if content.is_empty() {
                return Ok(vec![]);
            }
            Ok(vec![Finding {
                entity_label: self.0.to_string(),
                start: 0,
                end: content.len(),
                matched_text: Some(content.to_string()),
                score: None,
            }])
        }

        fn provider_type(&self) -> &'static str {
            self.0
        }
    }

    #[test]
    fn registry_insert_get_and_membership() {
        let mut registry = ProviderRegistry::new();
        assert!(registry.is_empty());

        let provider: Arc<dyn GuardrailProvider> = Arc::new(StubProvider("regex"));
        assert!(registry.insert("scanner", provider).is_none());

        assert_eq!(registry.len(), 1);
        assert!(registry.contains("scanner"));
        assert!(!registry.contains("missing"));

        let fetched = registry.get("scanner").expect("provider present");
        assert_eq!(fetched.provider_type(), "regex");
        assert!(registry.get("missing").is_none());
    }

    #[test]
    fn registry_insert_replaces_existing() {
        let mut registry = ProviderRegistry::new();
        registry.insert(
            "p",
            Arc::new(StubProvider("regex")) as Arc<dyn GuardrailProvider>,
        );
        let previous = registry.insert(
            "p",
            Arc::new(StubProvider("presidio")) as Arc<dyn GuardrailProvider>,
        );

        assert!(previous.is_some());
        assert_eq!(previous.unwrap().provider_type(), "regex");
        assert_eq!(registry.get("p").unwrap().provider_type(), "presidio");
        assert_eq!(registry.len(), 1);
    }

    #[tokio::test]
    async fn provider_analyze_returns_findings() {
        let provider = StubProvider("regex");
        assert!(provider.analyze("").await.unwrap().is_empty());

        let findings = provider.analyze("secret").await.unwrap();
        assert_eq!(findings.len(), 1);
        assert_eq!(findings[0].entity_label, "regex");
        assert_eq!(&"secret"[findings[0].start..findings[0].end], "secret");
    }

    /// Provider that always returns an error, to exercise failure-policy paths.
    struct FailingProvider;

    #[async_trait::async_trait]
    impl GuardrailProvider for FailingProvider {
        async fn analyze(&self, _content: &str) -> Result<Vec<Finding>, GuardrailProviderError> {
            Err(GuardrailProviderError::Unreachable("boom".to_string()))
        }

        fn provider_type(&self) -> &'static str {
            "failing"
        }
    }

    /// Provider that sleeps longer than any test timeout, to exercise the
    /// timeout branch.
    struct SlowProvider;

    #[async_trait::async_trait]
    impl GuardrailProvider for SlowProvider {
        async fn analyze(&self, _content: &str) -> Result<Vec<Finding>, GuardrailProviderError> {
            tokio::time::sleep(Duration::from_secs(3600)).await;
            Ok(vec![])
        }

        fn provider_type(&self) -> &'static str {
            "slow"
        }
    }

    #[tokio::test]
    async fn analyze_with_policy_success_returns_findings() {
        let provider = StubProvider("regex");
        let disposition = analyze_with_policy(
            &provider,
            "secret",
            Duration::from_secs(5),
            FailurePolicy::FailClose,
        )
        .await;

        match disposition {
            StageDisposition::Findings(findings) => {
                assert_eq!(findings.len(), 1);
                assert_eq!(findings[0].entity_label, "regex");
            }
            other => panic!("expected Findings, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn analyze_with_policy_error_fail_open_skips() {
        let disposition = analyze_with_policy(
            &FailingProvider,
            "content",
            Duration::from_secs(5),
            FailurePolicy::FailOpen,
        )
        .await;
        assert_eq!(disposition, StageDisposition::SkipOpen);
    }

    #[tokio::test]
    async fn analyze_with_policy_error_fail_close_halts() {
        let disposition = analyze_with_policy(
            &FailingProvider,
            "content",
            Duration::from_secs(5),
            FailurePolicy::FailClose,
        )
        .await;
        assert_eq!(disposition, StageDisposition::HaltClosed);
    }

    #[tokio::test]
    async fn analyze_with_policy_timeout_fail_open_skips() {
        let disposition = analyze_with_policy(
            &SlowProvider,
            "content",
            Duration::from_millis(10),
            FailurePolicy::FailOpen,
        )
        .await;
        assert_eq!(disposition, StageDisposition::SkipOpen);
    }

    #[tokio::test]
    async fn analyze_with_policy_timeout_fail_close_halts() {
        let disposition = analyze_with_policy(
            &SlowProvider,
            "content",
            Duration::from_millis(10),
            FailurePolicy::FailClose,
        )
        .await;
        assert_eq!(disposition, StageDisposition::HaltClosed);
    }

    #[tokio::test]
    async fn analyze_with_policy_accepts_arc_provider() {
        // The helper takes `&dyn`; an `Arc<dyn GuardrailProvider>` derefs to it,
        // matching how the engine holds shared provider instances.
        let provider: Arc<dyn GuardrailProvider> = Arc::new(StubProvider("regex"));
        let disposition = analyze_with_policy(
            provider.as_ref(),
            "secret",
            Duration::from_secs(5),
            FailurePolicy::FailOpen,
        )
        .await;
        assert!(matches!(disposition, StageDisposition::Findings(_)));
    }
}
