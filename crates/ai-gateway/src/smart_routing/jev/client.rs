//! Jev HTTP client: the shared System One client bound to the Jev family,
//! preserving the historical `JevClient::new` constructor and
/// `JevCallError` import path.

pub use crate::smart_routing::systemone::client::SystemOneCallError as JevCallError;

use std::fmt;
use std::time::Duration;

use crate::smart_routing::systemone::client::SystemOneClient;
use crate::smart_routing::systemone::models::{
    ModelsListing, SystemOneRequest, SystemOneResponse,
};

use super::FAMILY;

/// The Jev-bound System One HTTP client.
pub struct JevClient(SystemOneClient);

impl fmt::Debug for JevClient {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(formatter)
    }
}

impl JevClient {
    /// Build a client bound to one base URL and credential. `base_url`
    /// should already be validated (no query or fragment) by configuration
    /// validation; this constructor additionally enforces HTTPS.
    pub fn new(
        base_url: &str,
        api_key: &str,
        timeout: Duration,
        max_retries: u8,
        backoff: Duration,
    ) -> Result<Self, JevCallError> {
        SystemOneClient::new(FAMILY, base_url, api_key, timeout, max_retries, backoff).map(Self)
    }

    /// Evaluate a System One request.
    pub async fn evaluate(
        &self,
        request: &SystemOneRequest,
    ) -> Result<SystemOneResponse, JevCallError> {
        self.0.evaluate(request).await
    }

    /// List models available at the endpoint.
    pub async fn list_models(&self) -> Result<ModelsListing, JevCallError> {
        self.0.list_models().await
    }
}
