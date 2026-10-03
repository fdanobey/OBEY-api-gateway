//! Jev model discovery: re-exports of the shared System One discovery plus
//! the family-bound helper preserving the historical `latest_jev` contract.

pub use crate::smart_routing::systemone::discovery::{
    DiscoveryError, ModelsFetcher, ModelDiscovery, ResolutionOrigin, ResolvedModel,
};

use super::FAMILY;

/// Select the latest Jev-capable model from a listing of ids.
///
/// Version ordering: trailing semantic-version segments (`jev-1.13.0`,
/// `typesafe/jev-2.0.1`) compare numerically field-by-field; ids without a
/// parsable version compare lexicographically; versioned ids rank above
/// unversioned ones.
pub fn latest_jev(ids: &[String]) -> Option<String> {
    crate::smart_routing::systemone::discovery::latest_family_model(&FAMILY, ids)
}
