//! Sensitive-content screening for persistent-memory candidates.
//!
//! The scanner owns only precompiled local patterns and an optional shared
//! guardrail provider. It never stores or returns matched content.

use std::collections::BTreeSet;
use std::fmt;
use std::sync::{Arc, LazyLock};

use regex::Regex;

use crate::guardrail::provider::{GuardrailProvider, GuardrailProviderError};

static SECRET_PATTERNS: LazyLock<Vec<(&'static str, Regex)>> = LazyLock::new(|| {
    vec![
        (
            "openai_key",
            Regex::new(r"\bsk-[A-Za-z0-9]{20,}\b").expect("OpenAI key regex must compile"),
        ),
        (
            "aws_access_key",
            Regex::new(r"\bAKIA[A-Z0-9]{16}\b").expect("AWS access key regex must compile"),
        ),
        (
            "bearer_token",
            Regex::new(r"(?i)\bbearer\s+[A-Za-z0-9\-._~+/]{20,}={0,2}")
                .expect("Bearer token regex must compile"),
        ),
        (
            "url_password",
            Regex::new(r"(?i)\b[a-z][a-z0-9+.-]*://[^\s/@:]+:[^\s/@]+@[^\s/]+")
                .expect("URL password regex must compile"),
        ),
    ]
});

/// Metadata-only source of a sensitive-content finding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SensitiveMatchSource {
    SecretPattern,
    CustomPattern,
    PiiGuardrail,
}

/// Metadata-only result of scanning one memory candidate.
///
/// No matched values or byte offsets are retained.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SensitiveScanResult {
    pub contains_sensitive: bool,
    pub match_count: usize,
    pub sources: Vec<SensitiveMatchSource>,
    pub protection_bypassed: bool,
}

impl SensitiveScanResult {
    fn clear() -> Self {
        Self {
            contains_sensitive: false,
            match_count: 0,
            sources: Vec::new(),
            protection_bypassed: false,
        }
    }

    fn bypassed() -> Self {
        Self {
            protection_bypassed: true,
            ..Self::clear()
        }
    }
}

/// Handling for failures from the optional shared PII provider.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum PiiProviderFailurePolicy {
    /// Return a sanitized error so storage cannot continue accidentally.
    #[default]
    Surface,
    /// Ignore provider unavailability while retaining local-pattern findings.
    Skip,
}

/// Per-call controls for sensitive-content screening.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SensitiveScanOptions {
    pub allow_sensitive_storage: bool,
    pub pii_failure_policy: PiiProviderFailurePolicy,
}

/// Sanitized category for a PII provider failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PiiProviderErrorKind {
    Unreachable,
    UpstreamStatus,
    MalformedResponse,
    Analysis,
}

/// Errors produced while configuring or running a sensitive-content scan.
#[derive(Debug, thiserror::Error)]
pub enum SensitiveScanError {
    #[error("custom sensitive pattern at index {index} is invalid")]
    InvalidCustomPattern { index: usize },

    #[error("PII guardrail provider '{provider_type}' failed ({kind:?})")]
    PiiProvider {
        provider_type: &'static str,
        kind: PiiProviderErrorKind,
    },
}

/// Async scanner combining local secret patterns with an existing PII provider.
pub struct SensitiveContentScanner {
    custom_patterns: Vec<Regex>,
    pii_provider: Option<Arc<dyn GuardrailProvider>>,
}

impl SensitiveContentScanner {
    /// Compile custom patterns once and retain the active provider instance.
    pub fn new(
        custom_patterns: &[String],
        pii_provider: Option<Arc<dyn GuardrailProvider>>,
    ) -> Result<Self, SensitiveScanError> {
        let custom_patterns = custom_patterns
            .iter()
            .enumerate()
            .map(|(index, pattern)| {
                Regex::new(pattern).map_err(|_| SensitiveScanError::InvalidCustomPattern { index })
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self {
            custom_patterns,
            pii_provider,
        })
    }

    /// Scan with protection enabled and provider failures surfaced.
    pub async fn scan(&self, content: &str) -> Result<SensitiveScanResult, SensitiveScanError> {
        self.scan_with_options(content, SensitiveScanOptions::default())
            .await
    }

    /// Scan using explicit bypass and PII failure behavior.
    pub async fn scan_with_options(
        &self,
        content: &str,
        options: SensitiveScanOptions,
    ) -> Result<SensitiveScanResult, SensitiveScanError> {
        if options.allow_sensitive_storage {
            return Ok(SensitiveScanResult::bypassed());
        }

        let mut result = SensitiveScanResult::clear();
        let mut sources = BTreeSet::new();

        for (_, pattern) in SECRET_PATTERNS.iter() {
            let count = pattern.find_iter(content).count();
            if count > 0 {
                result.match_count += count;
                sources.insert(SensitiveMatchSource::SecretPattern);
            }
        }

        for pattern in &self.custom_patterns {
            let count = pattern.find_iter(content).count();
            if count > 0 {
                result.match_count += count;
                sources.insert(SensitiveMatchSource::CustomPattern);
            }
        }

        result.contains_sensitive = result.match_count > 0;
        result.sources = sources.iter().copied().collect();

        if result.contains_sensitive {
            return Ok(result);
        }

        if let Some(provider) = &self.pii_provider {
            match provider.analyze(content).await {
                Ok(findings) => {
                    if !findings.is_empty() {
                        result.match_count += findings.len();
                        sources.insert(SensitiveMatchSource::PiiGuardrail);
                    }
                }
                Err(error) => match options.pii_failure_policy {
                    PiiProviderFailurePolicy::Surface => {
                        return Err(SensitiveScanError::PiiProvider {
                            provider_type: provider.provider_type(),
                            kind: pii_error_kind(&error),
                        });
                    }
                    PiiProviderFailurePolicy::Skip => {}
                },
            }
        }

        result.contains_sensitive = result.match_count > 0;
        result.sources = sources.into_iter().collect();
        Ok(result)
    }
}

impl fmt::Debug for SensitiveContentScanner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("SensitiveContentScanner")
            .field("custom_pattern_count", &self.custom_patterns.len())
            .field(
                "pii_provider_type",
                &self
                    .pii_provider
                    .as_ref()
                    .map(|provider| provider.provider_type()),
            )
            .finish()
    }
}

fn pii_error_kind(error: &GuardrailProviderError) -> PiiProviderErrorKind {
    match error {
        GuardrailProviderError::Unreachable(_) => PiiProviderErrorKind::Unreachable,
        GuardrailProviderError::UpstreamStatus { .. } => PiiProviderErrorKind::UpstreamStatus,
        GuardrailProviderError::MalformedResponse(_) => PiiProviderErrorKind::MalformedResponse,
        GuardrailProviderError::Analysis(_) => PiiProviderErrorKind::Analysis,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use crate::guardrail::provider::Finding;

    use super::*;

    struct MockProvider {
        calls: AtomicUsize,
        response: MockResponse,
    }

    enum MockResponse {
        Clear,
        Finding,
        Failure,
    }

    impl MockProvider {
        fn new(response: MockResponse) -> Self {
            Self {
                calls: AtomicUsize::new(0),
                response,
            }
        }
    }

    #[async_trait::async_trait]
    impl GuardrailProvider for MockProvider {
        async fn analyze(&self, content: &str) -> Result<Vec<Finding>, GuardrailProviderError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            match self.response {
                MockResponse::Clear => Ok(Vec::new()),
                MockResponse::Finding => Ok(vec![Finding {
                    entity_label: "EMAIL_ADDRESS".to_owned(),
                    start: 0,
                    end: content.len(),
                    matched_text: Some(content.to_owned()),
                    score: Some(1.0),
                }]),
                MockResponse::Failure => {
                    Err(GuardrailProviderError::Unreachable(content.to_owned()))
                }
            }
        }

        fn provider_type(&self) -> &'static str {
            "mock_pii"
        }
    }

    fn scanner() -> SensitiveContentScanner {
        SensitiveContentScanner::new(&[], None).expect("scanner must build")
    }

    #[tokio::test]
    async fn detects_each_builtin_without_returning_values() {
        let cases = [
            "key sk-1234567890abcdefghij",
            "aws AKIA1234567890ABCDEF",
            "Authorization: Bearer abcdefghijklmnopqrstuvwx",
            "connect https://user:password@example.com/path",
        ];

        for content in cases {
            let result = scanner().scan(content).await.expect("scan succeeds");
            assert!(result.contains_sensitive);
            assert_eq!(result.match_count, 1);
            assert_eq!(result.sources, vec![SensitiveMatchSource::SecretPattern]);
            assert!(!format!("{result:?}").contains(content));
        }
    }

    #[tokio::test]
    async fn compiles_and_detects_custom_patterns() {
        let scanner = SensitiveContentScanner::new(&[r"PRIVATE-[0-9]{4}".to_owned()], None)
            .expect("custom pattern must compile");

        let result = scanner
            .scan("marker PRIVATE-1234")
            .await
            .expect("scan succeeds");

        assert!(result.contains_sensitive);
        assert_eq!(result.match_count, 1);
        assert_eq!(result.sources, vec![SensitiveMatchSource::CustomPattern]);
    }

    #[test]
    fn rejects_invalid_custom_patterns_without_echoing_them() {
        let error = SensitiveContentScanner::new(&["(?<secret".to_owned()], None)
            .expect_err("invalid pattern must fail");

        assert_eq!(
            error.to_string(),
            "custom sensitive pattern at index 0 is invalid"
        );
        assert!(!error.to_string().contains("secret"));
    }

    #[tokio::test]
    async fn reuses_shared_provider_and_reports_metadata_only() {
        let provider = Arc::new(MockProvider::new(MockResponse::Finding));
        let scanner =
            SensitiveContentScanner::new(&[], Some(provider.clone() as Arc<dyn GuardrailProvider>))
                .expect("scanner must build");
        let content = "person@example.com";

        let result = scanner.scan(content).await.expect("scan succeeds");

        assert_eq!(provider.calls.load(Ordering::SeqCst), 1);
        assert_eq!(result.match_count, 1);
        assert_eq!(result.sources, vec![SensitiveMatchSource::PiiGuardrail]);
        assert!(!format!("{result:?}").contains(content));
    }

    #[tokio::test]
    async fn bypass_avoids_local_and_provider_scans() {
        let provider = Arc::new(MockProvider::new(MockResponse::Finding));
        let scanner =
            SensitiveContentScanner::new(&[], Some(provider.clone() as Arc<dyn GuardrailProvider>))
                .expect("scanner must build");

        let result = scanner
            .scan_with_options(
                "sk-1234567890abcdefghij",
                SensitiveScanOptions {
                    allow_sensitive_storage: true,
                    ..SensitiveScanOptions::default()
                },
            )
            .await
            .expect("bypass succeeds");

        assert!(result.protection_bypassed);
        assert!(!result.contains_sensitive);
        assert_eq!(result.match_count, 0);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn provider_failures_are_sanitized_and_surface_by_default() {
        let provider = Arc::new(MockProvider::new(MockResponse::Failure));
        let scanner =
            SensitiveContentScanner::new(&[], Some(provider as Arc<dyn GuardrailProvider>))
                .expect("scanner must build");
        let content = "private provider payload";

        let error = scanner
            .scan(content)
            .await
            .expect_err("failure must surface");

        assert!(matches!(
            error,
            SensitiveScanError::PiiProvider {
                provider_type: "mock_pii",
                kind: PiiProviderErrorKind::Unreachable,
            }
        ));
        assert!(!error.to_string().contains(content));
    }

    #[tokio::test]
    async fn provider_failure_can_be_skipped_for_clear_local_content() {
        let provider = Arc::new(MockProvider::new(MockResponse::Failure));
        let scanner =
            SensitiveContentScanner::new(&[], Some(provider as Arc<dyn GuardrailProvider>))
                .expect("scanner must build");

        let result = scanner
            .scan_with_options(
                "ordinary preference",
                SensitiveScanOptions {
                    pii_failure_policy: PiiProviderFailurePolicy::Skip,
                    ..SensitiveScanOptions::default()
                },
            )
            .await
            .expect("provider failure is skipped");

        assert!(!result.contains_sensitive);
        assert_eq!(result.match_count, 0);
        assert!(result.sources.is_empty());
    }

    #[tokio::test]
    async fn local_findings_reject_without_calling_fallible_provider() {
        let provider = Arc::new(MockProvider::new(MockResponse::Failure));
        let scanner =
            SensitiveContentScanner::new(&[], Some(provider.clone() as Arc<dyn GuardrailProvider>))
                .expect("scanner must build");

        let result = scanner
            .scan("sk-1234567890abcdefghij")
            .await
            .expect("local positive finding rejects conservatively");

        assert!(result.contains_sensitive);
        assert_eq!(result.match_count, 1);
        assert_eq!(result.sources, vec![SensitiveMatchSource::SecretPattern]);
        assert_eq!(provider.calls.load(Ordering::SeqCst), 0);
    }

    #[tokio::test]
    async fn clear_content_returns_empty_result() {
        let provider = Arc::new(MockProvider::new(MockResponse::Clear));
        let scanner =
            SensitiveContentScanner::new(&[], Some(provider as Arc<dyn GuardrailProvider>))
                .expect("scanner must build");

        assert_eq!(
            scanner.scan("ordinary preference").await.unwrap(),
            SensitiveScanResult::clear()
        );
    }
}
