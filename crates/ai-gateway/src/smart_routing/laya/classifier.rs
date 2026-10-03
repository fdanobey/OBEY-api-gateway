//! The Laya classifier facade: the shared `SystemOneClassifier` bound to the
//! Laya family, constructed from [`LayaConfig`]. All behavior lives in the
//! shared `systemone` core.

use std::sync::Arc;

use crate::smart_routing::config::LayaConfig;
use crate::smart_routing::systemone::classifier::{
    SystemOneClassificationDetail, SystemOneClassifier, SystemOneEndpointSettings,
};
use crate::smart_routing::{ClassifierFailure, OptionalClassifier};

use super::FAMILY;

/// The Laya-bound System One classifier facade.
pub struct LayaClassifier {
    inner: SystemOneClassifier,
}

/// Extra detail produced by a successful Laya classification.
pub type LayaClassificationDetail = SystemOneClassificationDetail;

impl std::ops::Deref for LayaClassifier {
    type Target = SystemOneClassifier;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl LayaClassifier {
    /// Build from validated Laya configuration. The API key must already be
    /// resolved (empty keys are rejected by config validation).
    pub fn new(config: LayaConfig) -> Result<Self, ClassifierFailure> {
        let api_key = config
            .resolve_api_key()
            .ok_or(ClassifierFailure::Unavailable)?;
        let settings = SystemOneEndpointSettings {
            base_url: config.base_url.clone(),
            timeout_ms: config.timeout_ms,
            min_confidence: config.min_confidence,
            min_task_confidence: config.min_task_confidence,
            retry_max_attempts: config.retry.max_attempts,
            retry_backoff_ms: config.retry.backoff_ms,
            char_budget: config.char_budget,
            discovery_ttl_secs: config.discovery_ttl_secs,
            dimension_weights: config.dimension_weights,
            fallback_policy: config.fallback_policy,
            model: config.model.clone(),
        };
        Ok(Self {
            inner: SystemOneClassifier::new(FAMILY, settings, &api_key)?,
        })
    }

    /// Access the shared classifier (for `Arc` coercion into the
    /// `OptionalClassifier` trait object).
    pub fn into_inner(self) -> SystemOneClassifier {
        self.inner
    }

    /// Classify with full detail; the trait method is a thin wrapper.
    pub async fn classify_with_detail(
        &self,
        input: crate::smart_routing::ClassifierInput<'_>,
    ) -> Result<
        (
            crate::smart_routing::ClassifierOutput,
            LayaClassificationDetail,
        ),
        ClassifierFailure,
    > {
        self.inner.classify_with_detail(input).await
    }
}

impl From<LayaClassifier> for Arc<dyn OptionalClassifier> {
    fn from(classifier: LayaClassifier) -> Self {
        Arc::new(classifier)
    }
}

#[async_trait::async_trait]
impl OptionalClassifier for LayaClassifier {
    async fn classify(
        &self,
        input: crate::smart_routing::ClassifierInput<'_>,
    ) -> Result<crate::smart_routing::ClassifierOutput, ClassifierFailure> {
        self.inner.classify(input).await
    }
}
