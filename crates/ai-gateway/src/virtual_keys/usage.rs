//! Usage tracking and aggregation for virtual keys.
//!
//! The [`UsageTracker`] records per-request usage into the `virtual_key_usage`
//! table and advances the owning key's cumulative counters (spend, tokens,
//! request count, last-used timestamp), then answers time-ranged aggregation
//! queries. The SQL itself lives in [`KeyStore`]; this module coordinates the
//! insert + counter update and exposes the pure [`compute_cost`] helper.
//!
//! ## Missing token usage
//!
//! [`UsageTracker::record`] records exactly what it is given — it does not
//! parse provider responses. The decision of what to do when a provider
//! response omits token usage counts is made by the caller (integration task
//! 13.1): per Requirement 3.6 the caller may skip recording entirely and log a
//! warning, or per Requirement 4.5 estimate the token counts and record the
//! estimate. Either way, whatever [`UsageRecord`] reaches `record` is persisted
//! verbatim.

use std::sync::Arc;

use chrono::{DateTime, Utc};

use super::models::{UsageAggregate, UsageRecord};
use super::store::{KeyStore, KeyStoreError};

/// Records completed-request usage and answers aggregation queries.
///
/// Wraps the shared [`KeyStore`] so usage rows and key counters live in the
/// same `keys.db` database (Req 9.6).
pub struct UsageTracker {
    store: Arc<KeyStore>,
}

impl UsageTracker {
    /// Construct a tracker over the shared key store.
    pub fn new(store: Arc<KeyStore>) -> Self {
        Self { store }
    }

    /// Persist a single usage record and advance the key's cumulative counters.
    ///
    /// Inserts the per-request row into `virtual_key_usage` (Req 9.1), then
    /// advances the key's `current_spend_usd`, `current_tokens_used`,
    /// `request_count`, and `last_used_at` via
    /// [`KeyStore::update_usage_counters`] so budget enforcement sees the new
    /// totals (Req 3.5, 4.4). The token delta is `input_tokens + output_tokens`.
    ///
    /// This method records whatever it is given; handling of missing provider
    /// usage (skip vs estimate) is the caller's responsibility (see module docs,
    /// Req 3.6 / 4.5).
    ///
    /// _Requirements: 3.5, 4.4, 9.1, 9.6_
    pub fn record(&self, record: UsageRecord) -> Result<(), KeyStoreError> {
        self.store.insert_usage_record(&record)?;
        let tokens_delta = record.input_tokens.saturating_add(record.output_tokens) as i64;
        self.store
            .update_usage_counters(&record.key_id, record.cost_usd, tokens_delta)?;
        Ok(())
    }

    /// Aggregate usage for `key_id` over the inclusive range `[start, end]`.
    ///
    /// Returns summed spend and token counts plus the request count, or zero
    /// values when no records fall within the range (Req 9.2, 9.4).
    ///
    /// _Requirements: 9.2, 9.4_
    pub fn query_aggregate(
        &self,
        key_id: &str,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<UsageAggregate, KeyStoreError> {
        self.store.query_aggregate(key_id, start, end)
    }
}

/// Number of decimal places costs are rounded to (design "Cost Computation").
const COST_DECIMALS: f64 = 1_000_000.0;

/// Compute request cost in USD from token counts and per-million rates.
///
/// `cost = (input_tokens × input_rate / 1_000_000)
///        + (output_tokens × output_rate / 1_000_000)`, rounded to 6 decimal
/// places (Req 3.1, design "Cost Computation").
pub fn compute_cost(
    input_tokens: u64,
    output_tokens: u64,
    input_rate_per_million: f64,
    output_rate_per_million: f64,
) -> f64 {
    let input_cost = input_tokens as f64 * input_rate_per_million / COST_DECIMALS;
    let output_cost = output_tokens as f64 * output_rate_per_million / COST_DECIMALS;
    ((input_cost + output_cost) * COST_DECIMALS).round() / COST_DECIMALS
}

#[cfg(test)]
mod tests {
    use super::super::models::{CreateKeyParams, UsageQueryParams};
    use super::*;
    use crate::virtual_keys::VirtualKeyManager;
    use chrono::TimeZone;
    use tempfile::NamedTempFile;

    fn temp_manager() -> (VirtualKeyManager, NamedTempFile) {
        let temp = NamedTempFile::new().unwrap();
        let mgr = VirtualKeyManager::new(temp.path()).unwrap();
        (mgr, temp)
    }

    fn defaults() -> CreateKeyParams {
        CreateKeyParams {
            name: None,
            budget_limit_usd: None,
            token_budget: None,
            budget_window: None,
            requests_per_minute: None,
            tokens_per_minute: None,
            model_access: None,
            expires_in: None,
        }
    }

    fn ts(secs: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(secs, 0).unwrap()
    }

    fn record_at(key_id: &str, secs: i64, input: u64, output: u64, cost: f64) -> UsageRecord {
        UsageRecord {
            key_id: key_id.to_string(),
            model_group: "grp".to_string(),
            model: "gpt-4".to_string(),
            input_tokens: input,
            output_tokens: output,
            cost_usd: cost,
            timestamp: ts(secs),
        }
    }

    /// Recording usage sums cost/tokens/requests in aggregation and advances the
    /// key's cumulative counters (Req 9.1, 9.2, 3.5, 4.4).
    #[tokio::test]
    async fn record_then_query_aggregate_sums() {
        let (mgr, _tmp) = temp_manager();
        let key = mgr.create_key(defaults()).await.unwrap();
        let tracker = UsageTracker::new(std::sync::Arc::clone(&mgr.store));

        tracker
            .record(record_at(&key.id, 1_000, 100, 50, 0.0015))
            .unwrap();
        tracker
            .record(record_at(&key.id, 2_000, 200, 25, 0.0020))
            .unwrap();

        let agg = tracker
            .query_aggregate(&key.id, ts(0), ts(10_000))
            .unwrap();
        assert!((agg.total_spend_usd - 0.0035).abs() < 1e-9);
        assert_eq!(agg.total_input_tokens, 300);
        assert_eq!(agg.total_output_tokens, 75);
        assert_eq!(agg.total_requests, 2);

        // Cumulative counters on the key advanced (Req 3.5, 4.4).
        let stored = mgr.store.get_key_by_id(&key.id).unwrap().unwrap();
        assert!((stored.current_spend_usd - 0.0035).abs() < 1e-9);
        assert_eq!(stored.current_tokens_used, 375);
        assert_eq!(stored.request_count, 2);
        assert!(stored.last_used_at.is_some());
    }

    /// Aggregation bounds are inclusive on both ends (Req 9.2).
    #[tokio::test]
    async fn query_aggregate_time_range_inclusive() {
        let (mgr, _tmp) = temp_manager();
        let key = mgr.create_key(defaults()).await.unwrap();
        let tracker = UsageTracker::new(std::sync::Arc::clone(&mgr.store));

        tracker.record(record_at(&key.id, 100, 10, 0, 0.01)).unwrap();
        tracker.record(record_at(&key.id, 200, 20, 0, 0.02)).unwrap();
        tracker.record(record_at(&key.id, 300, 30, 0, 0.03)).unwrap();

        // Range [100, 300] includes all three (both boundaries inclusive).
        let all = tracker.query_aggregate(&key.id, ts(100), ts(300)).unwrap();
        assert_eq!(all.total_requests, 3);

        // Range [200, 200] includes only the boundary record.
        let one = tracker.query_aggregate(&key.id, ts(200), ts(200)).unwrap();
        assert_eq!(one.total_requests, 1);
        assert_eq!(one.total_input_tokens, 20);

        // Range excluding endpoints.
        let mid = tracker.query_aggregate(&key.id, ts(150), ts(250)).unwrap();
        assert_eq!(mid.total_requests, 1);
        assert_eq!(mid.total_input_tokens, 20);
    }

    /// A range with no matching records yields all-zero aggregates (Req 9.4).
    #[tokio::test]
    async fn query_aggregate_empty_range_returns_zeros() {
        let (mgr, _tmp) = temp_manager();
        let key = mgr.create_key(defaults()).await.unwrap();
        let tracker = UsageTracker::new(std::sync::Arc::clone(&mgr.store));

        tracker.record(record_at(&key.id, 5_000, 10, 5, 0.01)).unwrap();

        let agg = tracker.query_aggregate(&key.id, ts(0), ts(1_000)).unwrap();
        assert_eq!(agg.total_spend_usd, 0.0);
        assert_eq!(agg.total_input_tokens, 0);
        assert_eq!(agg.total_output_tokens, 0);
        assert_eq!(agg.total_requests, 0);
    }

    /// Querying usage for an unknown key id returns `KeyError::NotFound` (Req 9.3).
    #[tokio::test]
    async fn query_usage_unknown_id_not_found() {
        let (mgr, _tmp) = temp_manager();
        let params = UsageQueryParams {
            start: ts(0),
            end: ts(10_000),
        };
        let err = mgr.query_usage("does-not-exist", params).await.unwrap_err();
        assert!(
            matches!(err, crate::virtual_keys::KeyError::NotFound(_)),
            "expected NotFound, got {err:?}"
        );
    }

    /// `query_usage` returns aggregates for an existing key (Req 9.2).
    #[tokio::test]
    async fn query_usage_existing_key_aggregates() {
        let (mgr, _tmp) = temp_manager();
        let key = mgr.create_key(defaults()).await.unwrap();
        mgr.record_usage(record_at(&key.id, 100, 40, 60, 0.5))
            .await
            .unwrap();

        let params = UsageQueryParams {
            start: ts(0),
            end: ts(1_000),
        };
        let agg = mgr.query_usage(&key.id, params).await.unwrap();
        assert_eq!(agg.total_input_tokens, 40);
        assert_eq!(agg.total_output_tokens, 60);
        assert_eq!(agg.total_requests, 1);
        assert!((agg.total_spend_usd - 0.5).abs() < 1e-9);
    }

    /// `compute_cost` applies the formula and rounds to 6 decimal places (Req 3.1).
    #[test]
    fn compute_cost_rounds_to_six_decimals() {
        // 1000 in @ $3/M + 500 out @ $6/M = 0.003 + 0.003 = 0.006.
        let cost = compute_cost(1_000, 500, 3.0, 6.0);
        assert!((cost - 0.006).abs() < 1e-12);

        // Rounding: 1 token @ $1/M = 0.000001 exactly.
        assert!((compute_cost(1, 0, 1.0, 0.0) - 0.000001).abs() < 1e-12);

        // Sub-micro-dollar amounts round to 6 decimals.
        // 1 token @ $0.4/M = 0.0000004 -> rounds to 0.0.
        assert_eq!(compute_cost(1, 0, 0.4, 0.0), 0.0);
        // 1 token @ $1.5/M = 0.0000015 -> rounds to 0.000002 (round half away).
        let rounded = compute_cost(1, 0, 1.5, 0.0);
        assert!((rounded - 0.000002).abs() < 1e-12, "got {rounded}");

        // Zero tokens -> zero cost.
        assert_eq!(compute_cost(0, 0, 10.0, 20.0), 0.0);
    }

    use proptest::prelude::*;

    /// A fresh tokio runtime per proptest case, used to drive the async
    /// `create_key` before invoking the synchronous `record`/`query_aggregate`.
    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Runtime::new().unwrap()
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 100,
            .. ProptestConfig::default()
        })]

        // Feature: virtual-key-management, Property 15: Usage Aggregation
        //
        // For any set of usage records for a key and any time range
        // [start, end], the aggregated spend SHALL equal the sum of `cost_usd`
        // for records with timestamps within [start, end] inclusive, and the
        // aggregated token counts SHALL equal the sum of respective
        // `input_tokens` and `output_tokens` for records in that range.
        //
        // **Validates: Requirements 9.1, 9.2**

        /// Insert a random set of records for one key, then assert that
        /// `query_aggregate` over a random inclusive range equals the
        /// independently-computed sums/counts over the in-range subset.
        ///
        /// Costs are generated as integer micro-dollars and converted via
        /// `micros / 1e6` so the reference sum stays exact; the f64 spend is
        /// still compared with a small epsilon to absorb SQLite REAL summation
        /// error. Timestamps are whole seconds (the store truncates to
        /// `.timestamp()`), so range boundaries land exactly.
        #[test]
        fn prop_usage_aggregation_matches_in_range_sums(
            records in prop::collection::vec(
                (0i64..=10_000i64, 0u64..=100_000u64, 0u64..=100_000u64, 0u64..=10_000_000u64),
                0..=30,
            ),
            start_secs in 0i64..=10_000i64,
            end_secs in 0i64..=10_000i64,
        ) {
            let (lo, hi) = if start_secs <= end_secs {
                (start_secs, end_secs)
            } else {
                (end_secs, start_secs)
            };

            // Independently compute the expected aggregate over the in-range
            // subset (timestamp in [lo, hi] inclusive).
            let mut exp_micros: u128 = 0;
            let mut exp_input: u64 = 0;
            let mut exp_output: u64 = 0;
            let mut exp_requests: u64 = 0;
            for (secs, input, output, micros) in &records {
                if *secs >= lo && *secs <= hi {
                    exp_micros += *micros as u128;
                    exp_input += *input;
                    exp_output += *output;
                    exp_requests += 1;
                }
            }
            let exp_spend = exp_micros as f64 / 1_000_000.0;

            let agg = rt().block_on(async {
                let (mgr, _tmp) = temp_manager();
                let key = mgr.create_key(defaults()).await.unwrap();
                let tracker = UsageTracker::new(std::sync::Arc::clone(&mgr.store));
                for (secs, input, output, micros) in &records {
                    let cost = *micros as f64 / 1_000_000.0;
                    tracker
                        .record(record_at(&key.id, *secs, *input, *output, cost))
                        .unwrap();
                }
                tracker.query_aggregate(&key.id, ts(lo), ts(hi)).unwrap()
            });

            prop_assert!(
                (agg.total_spend_usd - exp_spend).abs() < 1e-6,
                "spend mismatch: got {} expected {} (range [{}, {}])",
                agg.total_spend_usd,
                exp_spend,
                lo,
                hi
            );
            prop_assert_eq!(agg.total_input_tokens, exp_input);
            prop_assert_eq!(agg.total_output_tokens, exp_output);
            prop_assert_eq!(agg.total_requests, exp_requests);
        }
    }
}
