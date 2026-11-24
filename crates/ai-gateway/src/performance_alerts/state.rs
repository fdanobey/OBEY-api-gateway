//! Pure drift-evaluation core for performance alerts (spec:
//! performance-alerts, Component 1).
//!
//! Decides, from caller-supplied daily ledger aggregates, whether a
//! (scope, signal, day) finding fires for the last complete UTC day
//! against a 7-day rolling baseline. The module is deliberately
//! pure — no I/O, no async, no clocks: identical inputs always yield
//! identical findings and skips (the `metrics/forecast.rs`
//! discipline). Baselines are request-weighted where a mean is
//! involved: latency and error rate divide period sums, never an
//! average of daily ratios.
//!
//! Fixed thresholds and minimum-data gates (design table, adopted
//! from LiteLLM's shipped defaults):
//!
//! | Signal     | Ratio              | Absolute floor | Skip when                                                     |
//! |------------|--------------------|----------------|---------------------------------------------------------------|
//! | spend      | ≥3x daily baseline | ≥$50           | baseline spend = 0, or <4 traffic days                        |
//! | traffic    | ≥10x daily baseline | ≥1000 req     | <4 traffic days                                               |
//! | latency    | ≥3x baseline mean  | ≥500ms         | baseline mean = 0, or <4 traffic days, or <100 req on D      |
//! | error_rate | ≥3x baseline rate  | ≥2pp           | baseline rate = 0, or <4 traffic days, or <500 req on D      |
//!
//! _Requirements: 1.1, 1.2, 2.1–2.8, 3.1, 3.2_

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

/// Minimum baseline days with traffic before any signal is trusted
/// (design "Skip when": traffic on fewer than 4 of the 7 baseline
/// days).
pub const MIN_BASELINE_TRAFFIC_DAYS: usize = 4;

/// Spend fires at ≥3x the daily baseline with ≥$50 absolute drift.
/// Integer, cross-multiplied against the 7-day baseline sum in
/// `evaluate` (exact thresholds — no float drift).
pub const SPEND_RATIO: i128 = 3;
/// Minimum absolute spend drift: $50, in micro-USD.
pub const SPEND_MIN_DELTA_MICRO_USD: i128 = 50_000_000;

/// Traffic fires at ≥10x the daily baseline with ≥1000 absolute
/// requests (integer cross-multiplication).
pub const TRAFFIC_RATIO: i128 = 10;
/// Minimum absolute traffic drift: 1000 requests.
pub const TRAFFIC_MIN_DELTA: i128 = 1_000;

/// Latency fires at ≥3x the request-weighted baseline mean with
/// ≥500ms absolute drift (integer cross-multiplication).
pub const LATENCY_RATIO: i128 = 3;
/// Minimum absolute mean-latency drift: 500ms.
pub const LATENCY_MIN_DELTA_MS: i128 = 500;
/// Minimum requests on the evaluated day before latency is trusted.
pub const LATENCY_MIN_DAY_REQUESTS: i64 = 100;

/// Error rate fires at ≥3x the request-weighted baseline rate with
/// ≥2 percentage points absolute drift (integer
/// cross-multiplication; the pp delta compares ×100-scaled integer
/// cross-products, so 2pp stays exact).
pub const ERROR_RATE_RATIO: i128 = 3;
/// Minimum absolute error-rate drift: 2 percentage points.
pub const ERROR_RATE_MIN_DELTA_PP: i128 = 2;
/// Minimum requests on the evaluated day before error rate is
/// trusted.
pub const ERROR_RATE_MIN_DAY_REQUESTS: i64 = 500;

/// Baseline window length (days): the cross-multiplication
/// denominator that replaces the per-day average in fire decisions.
const BASELINE_WINDOW_DAYS: i128 = 7;
/// Scale factoring percentage points into integers (2pp = 2/100).
const PERCENT_SCALE: i128 = 100;

/// Micro-USD per USD for the spend DISPLAY values only — fire/no-fire
/// decisions use the integer micro-USD sums directly.
const MICRO_USD_PER_USD: f64 = 1_000_000.0;

/// One day of ledger aggregates for one scope (design Data Models;
/// `duration_ms` is NOT NULL in the requests ledger, so every row is
/// timed and `requests` doubles as the timed-row count).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DailyStats {
    /// UTC day the row buckets.
    pub day: NaiveDate,
    /// Σ cost in micro-USD for the day.
    pub cost_micro_usd: i64,
    /// Σ finished requests for the day.
    pub requests: i64,
    /// Σ failed requests (status_code ≥ 400) for the day.
    pub errors: i64,
    /// Σ upstream duration in milliseconds for the day.
    pub duration_ms_sum: i64,
}

/// Evaluation dimension a [`DailyStats`] row belongs to (scope keys
/// are provider or model-group NAMES only — no ids; the gateway
/// scope's name is the literal `"gateway"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Scope {
    Gateway,
    Provider,
    ModelGroup,
}

impl Scope {
    /// Serde-friendly discriminant for payloads and ledger columns.
    pub fn as_str(&self) -> &'static str {
        match self {
            Scope::Gateway => "gateway",
            Scope::Provider => "provider",
            Scope::ModelGroup => "model_group",
        }
    }
}

/// Drift signal under evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Signal {
    Spend,
    Traffic,
    ErrorRate,
    Latency,
}

impl Signal {
    /// Serde-friendly discriminant for payloads.
    pub fn as_str(&self) -> &'static str {
        match self {
            Signal::Spend => "spend",
            Signal::Traffic => "traffic",
            Signal::ErrorRate => "error_rate",
            Signal::Latency => "latency",
        }
    }
}

/// Unit accompanying a [`Finding`]'s values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Unit {
    Usd,
    Requests,
    Ratio,
    Ms,
}

/// A fired (scope, signal, day) comparison: observed vs baseline with
/// the fixed condition that fired.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Finding {
    /// Signal that fired.
    pub signal: Signal,
    /// Observed day's value for the signal.
    pub observed_value: f64,
    /// Baseline (7-day) value for the signal.
    pub baseline_value: f64,
    /// Unit both values share.
    pub unit: Unit,
    /// Human-readable fixed condition that fired.
    pub threshold: &'static str,
}

/// A minimum-data gate that suppressed one signal's check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkipReason {
    /// Traffic on fewer than 4 of the 7 baseline days.
    InsufficientBaselineDays,
    /// Baseline spend is zero (spend signal only).
    ZeroBaselineSpend,
    /// Request-weighted baseline mean duration is zero (latency
    /// signal only).
    ZeroBaselineLatency,
    /// Baseline request-weighted error rate is zero (error_rate
    /// signal only).
    ZeroBaselineErrorRate,
    /// Zero requests on the evaluated day.
    NoTrafficOnEvaluatedDay,
    /// Fewer than 100 requests on the evaluated day (latency).
    TooFewRequestsForLatency,
    /// Fewer than 500 requests on the evaluated day (error_rate).
    TooFewRequestsForErrorRate,
}

/// Evaluate the last complete UTC day `observed` against the 7-day
/// `baseline` for one scope: findings for every fired signal plus
/// every minimum-data skip (Req 2.8). Pure — same inputs always
/// yield the same output.
pub fn evaluate(
    scope: &Scope,
    day: NaiveDate,
    baseline: &[DailyStats; 7],
    observed: &DailyStats,
) -> (Vec<Finding>, Vec<SkipReason>) {
    let _ = scope;
    let mut findings = Vec::new();
    let mut skips = Vec::new();
    let traffic_days = baseline.iter().filter(|d| d.requests > 0).count();
    let enough_days = traffic_days >= MIN_BASELINE_TRAFFIC_DAYS;
    let day_requests = if observed.day == day {
        observed.requests
    } else {
        0
    };

    if day_requests <= 0 {
        skips.push(SkipReason::NoTrafficOnEvaluatedDay);
        if !enough_days {
            skips.push(SkipReason::InsufficientBaselineDays);
        }
        return (findings, skips);
    }
    if !enough_days {
        skips.push(SkipReason::InsufficientBaselineDays);
    }

    // Integer decision math (drift-free thresholds): every fire
    // comparison is an i128 cross-multiplication over the raw i64
    // ledger sums, never a float division. All inputs are
    // non-negative ledger aggregates; the practical bounds
    // (≤ ~9.2e18 per field) keep every product far below the i128
    // limit (a product only overflows past ~1.7e19 per factor, i.e.
    // beyond any real ledger magnitude). The f64 values on the
    // findings are DISPLAY-ONLY, computed with safe division.
    let baseline_cost_micro: i128 =
        baseline.iter().map(|d| d.cost_micro_usd as i128).sum();
    let observed_cost_micro: i128 = observed.cost_micro_usd as i128;
    if enough_days {
        if baseline_cost_micro <= 0 {
            skips.push(SkipReason::ZeroBaselineSpend);
        } else if observed_cost_micro * BASELINE_WINDOW_DAYS >= SPEND_RATIO * baseline_cost_micro
            && observed_cost_micro * BASELINE_WINDOW_DAYS - baseline_cost_micro
                >= SPEND_MIN_DELTA_MICRO_USD * BASELINE_WINDOW_DAYS
        {
            findings.push(Finding {
                signal: Signal::Spend,
                observed_value: observed_cost_micro as f64 / MICRO_USD_PER_USD,
                baseline_value: baseline_cost_micro as f64
                    / MICRO_USD_PER_USD
                    / BASELINE_WINDOW_DAYS as f64,
                unit: Unit::Usd,
                threshold: ">=3x baseline and >=$50 over",
            });
        }
    }

    let baseline_requests_total: i128 = baseline.iter().map(|d| d.requests as i128).sum();
    let observed_requests: i128 = observed.requests as i128;
    if enough_days
        && observed_requests * BASELINE_WINDOW_DAYS >= TRAFFIC_RATIO * baseline_requests_total
        && observed_requests * BASELINE_WINDOW_DAYS - baseline_requests_total
            >= TRAFFIC_MIN_DELTA * BASELINE_WINDOW_DAYS
    {
        findings.push(Finding {
            signal: Signal::Traffic,
            observed_value: observed_requests as f64,
            baseline_value: baseline_requests_total as f64 / BASELINE_WINDOW_DAYS as f64,
            unit: Unit::Requests,
            threshold: ">=10x baseline and >=1000 over",
        });
    }

    let baseline_duration_sum: i128 = baseline.iter().map(|d| d.duration_ms_sum as i128).sum();
    if enough_days {
        let baseline_has_requests = baseline_requests_total > 0;
        let observed_duration: i128 = observed.duration_ms_sum as i128;
        if baseline_has_requests && baseline_duration_sum <= 0 {
            skips.push(SkipReason::ZeroBaselineLatency);
        } else if day_requests < LATENCY_MIN_DAY_REQUESTS {
            skips.push(SkipReason::TooFewRequestsForLatency);
        } else if baseline_has_requests
            && observed_duration * baseline_requests_total
                >= LATENCY_RATIO * baseline_duration_sum * observed_requests
            && observed_duration * baseline_requests_total
                - baseline_duration_sum * observed_requests
                >= LATENCY_MIN_DELTA_MS * observed_requests * baseline_requests_total
        {
            findings.push(Finding {
                signal: Signal::Latency,
                observed_value: observed_duration as f64 / observed_requests as f64,
                baseline_value: baseline_duration_sum as f64 / baseline_requests_total as f64,
                unit: Unit::Ms,
                threshold: ">=3x baseline and >=500ms over",
            });
        }
    }

    let baseline_errors_total: i128 = baseline.iter().map(|d| d.errors as i128).sum();
    let observed_errors: i128 = observed.errors as i128;
    if enough_days {
        if baseline_requests_total > 0 && baseline_errors_total <= 0 {
            skips.push(SkipReason::ZeroBaselineErrorRate);
        } else if day_requests < ERROR_RATE_MIN_DAY_REQUESTS {
            skips.push(SkipReason::TooFewRequestsForErrorRate);
        } else if baseline_requests_total > 0
            && observed_errors * baseline_requests_total
                >= ERROR_RATE_RATIO * baseline_errors_total * observed_requests
            && (observed_errors * baseline_requests_total
                - baseline_errors_total * observed_requests)
                * PERCENT_SCALE
                >= ERROR_RATE_MIN_DELTA_PP * observed_requests * baseline_requests_total
        {
            findings.push(Finding {
                signal: Signal::ErrorRate,
                observed_value: observed_errors as f64 / observed_requests as f64,
                baseline_value: baseline_errors_total as f64 / baseline_requests_total as f64,
                unit: Unit::Ratio,
                threshold: ">=3x baseline and >=2pp over",
            });
        }
    }

    (findings, skips)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn day(n: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 11, n).unwrap()
    }

    fn stats(n: u32, cost_micro_usd: i64, requests: i64, errors: i64, duration_ms_sum: i64) -> DailyStats {
        DailyStats {
            day: day(n),
            cost_micro_usd,
            requests,
            errors,
            duration_ms_sum,
        }
    }

    fn flat_baseline(
        cost_micro_usd: i64,
        requests: i64,
        errors: i64,
        duration_ms_sum: i64,
    ) -> [DailyStats; 7] {
        [
            stats(1, cost_micro_usd, requests, errors, duration_ms_sum),
            stats(2, cost_micro_usd, requests, errors, duration_ms_sum),
            stats(3, cost_micro_usd, requests, errors, duration_ms_sum),
            stats(4, cost_micro_usd, requests, errors, duration_ms_sum),
            stats(5, cost_micro_usd, requests, errors, duration_ms_sum),
            stats(6, cost_micro_usd, requests, errors, duration_ms_sum),
            stats(7, cost_micro_usd, requests, errors, duration_ms_sum),
        ]
    }

    fn signals(findings: &[Finding]) -> Vec<Signal> {
        findings.iter().map(|f| f.signal).collect()
    }

    #[test]
    fn scope_and_signal_serde_names() {
        assert_eq!(Scope::Gateway.as_str(), "gateway");
        assert_eq!(Scope::Provider.as_str(), "provider");
        assert_eq!(Scope::ModelGroup.as_str(), "model_group");
        assert_eq!(
            serde_json::to_value(Scope::ModelGroup).unwrap(),
            serde_json::json!("model_group")
        );
        assert_eq!(Signal::ErrorRate.as_str(), "error_rate");
        assert_eq!(
            serde_json::to_value(Signal::Spend).unwrap(),
            serde_json::json!("spend")
        );
        assert_eq!(
            serde_json::to_value(Signal::ErrorRate).unwrap(),
            serde_json::json!("error_rate")
        );
        assert_eq!(
            serde_json::to_value(Unit::Usd).unwrap(),
            serde_json::json!("usd")
        );
        assert_eq!(
            serde_json::to_value(SkipReason::TooFewRequestsForLatency).unwrap(),
            serde_json::json!("too_few_requests_for_latency")
        );
    }

    #[test]
    fn spend_fires_exactly_at_thresholds() {
        // $25/day ⇒ baseline $25; observed $75 = exactly 3x with
        // exactly $50 delta.
        let baseline = flat_baseline(25_000_000, 1_000, 10, 250_000);
        let observed = stats(8, 75_000_000, 1_000, 10, 250_000);
        let (findings, skips) = evaluate(&Scope::Gateway, day(8), &baseline, &observed);
        assert_eq!(signals(&findings), vec![Signal::Spend]);
        assert!(skips.is_empty());
        assert_eq!(findings[0].observed_value, 75.0);
        assert_eq!(findings[0].baseline_value, 25.0);
        assert_eq!(findings[0].unit, Unit::Usd);
        assert_eq!(findings[0].threshold, ">=3x baseline and >=$50 over");
    }

    #[test]
    fn spend_below_ratio_does_not_fire() {
        // $30/day baseline; $89.99 observed: delta $59.99 ≥ $50 but
        // ratio 2.9997 < 3.
        let baseline = flat_baseline(30_000_000, 1_000, 10, 250_000);
        let observed = stats(8, 89_990_000, 1_000, 10, 250_000);
        let (findings, skips) = evaluate(&Scope::Provider, day(8), &baseline, &observed);
        assert!(findings.is_empty());
        assert!(skips.is_empty());
    }

    #[test]
    fn spend_below_delta_does_not_fire() {
        // $10/day baseline; $50 observed: ratio 5x but delta $40
        // < $50.
        let baseline = flat_baseline(10_000_000, 1_000, 10, 250_000);
        let observed = stats(8, 50_000_000, 1_000, 10, 250_000);
        let (findings, skips) = evaluate(&Scope::ModelGroup, day(8), &baseline, &observed);
        assert!(findings.is_empty());
        assert!(skips.is_empty());
    }

    #[test]
    fn traffic_fires_exactly_at_thresholds() {
        // 200/day ⇒ baseline 200; 2000 observed = exactly 10x with
        // 1800 ≥ 1000 delta.
        let baseline = flat_baseline(1_000_000, 200, 2, 20_000);
        let observed = stats(8, 1_000_000, 2_000, 20, 200_000);
        let (findings, skips) = evaluate(&Scope::Gateway, day(8), &baseline, &observed);
        assert_eq!(signals(&findings), vec![Signal::Traffic]);
        assert!(skips.is_empty());
        assert_eq!(findings[0].observed_value, 2_000.0);
        assert_eq!(findings[0].baseline_value, 200.0);
        assert_eq!(findings[0].unit, Unit::Requests);
        assert_eq!(findings[0].threshold, ">=10x baseline and >=1000 over");
    }

    #[test]
    fn traffic_below_ratio_does_not_fire() {
        // 1999 observed: delta 1799 ≥ 1000 but ratio 9.995 < 10.
        let baseline = flat_baseline(1_000_000, 200, 2, 20_000);
        let observed = stats(8, 1_000_000, 1_999, 19, 199_900);
        let (findings, skips) = evaluate(&Scope::Gateway, day(8), &baseline, &observed);
        assert!(findings.is_empty());
        assert!(skips.is_empty());
    }

    #[test]
    fn traffic_below_delta_does_not_fire() {
        // 60/day ⇒ baseline 60; 600 observed = exactly 10x but delta
        // 540 < 1000.
        let baseline = flat_baseline(1_000_000, 60, 1, 6_000);
        let observed = stats(8, 1_000_000, 600, 10, 60_000);
        let (findings, skips) = evaluate(&Scope::Gateway, day(8), &baseline, &observed);
        assert!(findings.is_empty());
        assert!(skips.is_empty());
    }

    #[test]
    fn latency_fires_exactly_at_thresholds() {
        // Baseline mean 250ms; observed mean 750ms = exactly 3x with
        // exactly 500ms delta.
        let baseline = flat_baseline(1_000_000, 1_000, 10, 250_000);
        let observed = stats(8, 1_000_000, 500, 5, 375_000);
        let (findings, skips) = evaluate(&Scope::Provider, day(8), &baseline, &observed);
        assert_eq!(signals(&findings), vec![Signal::Latency]);
        assert!(skips.is_empty());
        assert_eq!(findings[0].observed_value, 750.0);
        assert_eq!(findings[0].baseline_value, 250.0);
        assert_eq!(findings[0].unit, Unit::Ms);
        assert_eq!(findings[0].threshold, ">=3x baseline and >=500ms over");
    }

    #[test]
    fn latency_below_ratio_does_not_fire() {
        // Baseline mean 400ms; observed mean 1199ms: delta 799 ≥ 500
        // but ratio 2.9975 < 3.
        let baseline = flat_baseline(1_000_000, 1_000, 10, 400_000);
        let observed = stats(8, 1_000_000, 500, 5, 599_500);
        let (findings, skips) = evaluate(&Scope::Provider, day(8), &baseline, &observed);
        assert!(findings.is_empty());
        assert!(skips.is_empty());
    }

    #[test]
    fn latency_below_delta_does_not_fire() {
        // Baseline mean 40ms; observed mean 289ms: ratio 7.2x ≥ 3 but
        // delta 249ms < 500ms.
        let baseline = flat_baseline(1_000_000, 1_000, 10, 40_000);
        let observed = stats(8, 1_000_000, 500, 5, 144_500);
        let (findings, skips) = evaluate(&Scope::Provider, day(8), &baseline, &observed);
        assert!(findings.is_empty());
        assert!(skips.is_empty());
    }

    #[test]
    fn error_rate_fires_exactly_at_thresholds() {
        // Dyadic-exact rates: baseline 102/1024 = 0.099609375;
        // observed 153/512 = 0.298828125 = exactly 3x with delta
        // 0.19921875 ≥ 2pp.
        let baseline = flat_baseline(1_000_000, 1_024, 102, 102_400);
        let observed = stats(8, 1_000_000, 512, 153, 51_200);
        let (findings, skips) = evaluate(&Scope::ModelGroup, day(8), &baseline, &observed);
        assert_eq!(signals(&findings), vec![Signal::ErrorRate]);
        assert!(skips.is_empty());
        assert_eq!(findings[0].observed_value, 0.298828125);
        assert_eq!(findings[0].baseline_value, 0.099609375);
        assert_eq!(findings[0].unit, Unit::Ratio);
        assert_eq!(findings[0].threshold, ">=3x baseline and >=2pp over");
    }

    #[test]
    fn error_rate_below_ratio_does_not_fire() {
        // 305/1024 = 0.2978515625 < 3x of 0.099609375 despite delta
        // 0.1982421875 ≥ 2pp.
        let baseline = flat_baseline(1_000_000, 1_024, 102, 102_400);
        let observed = stats(8, 1_000_000, 1_024, 305, 102_400);
        let (findings, skips) = evaluate(&Scope::ModelGroup, day(8), &baseline, &observed);
        assert!(findings.is_empty());
        assert!(skips.is_empty());
    }

    #[test]
    fn error_rate_below_delta_does_not_fire() {
        // Baseline 2/1024; observed 20/1024: ratio 10x ≥ 3 but delta
        // 0.017578125 < 2pp.
        let baseline = flat_baseline(1_000_000, 1_024, 2, 102_400);
        let observed = stats(8, 1_000_000, 1_024, 20, 102_400);
        let (findings, skips) = evaluate(&Scope::ModelGroup, day(8), &baseline, &observed);
        assert!(findings.is_empty());
        assert!(skips.is_empty());
    }

    #[test]
    fn fewer_than_four_baseline_traffic_days_skips_everything() {
        // Only 3 of 7 baseline days have traffic.
        let mut baseline = flat_baseline(20_000_000, 1_000, 10, 250_000);
        for d in baseline.iter_mut().take(4) {
            d.requests = 0;
            d.cost_micro_usd = 0;
            d.errors = 0;
            d.duration_ms_sum = 0;
        }
        let observed = stats(8, 60_000_000, 2_000, 100, 250_000);
        let (findings, skips) = evaluate(&Scope::Gateway, day(8), &baseline, &observed);
        assert!(findings.is_empty());
        assert_eq!(skips, vec![SkipReason::InsufficientBaselineDays]);
    }

    #[test]
    fn exactly_four_baseline_traffic_days_passes_the_gate() {
        let mut baseline = flat_baseline(20_000_000, 1_000, 10, 250_000);
        for d in baseline.iter_mut().take(3) {
            d.requests = 0;
            d.cost_micro_usd = 0;
            d.errors = 0;
            d.duration_ms_sum = 0;
        }
        // Baseline spend = 4 × $20 / 7 ≈ $11.43; $60 observed passes
        // the ratio but the $48.57 delta misses $50 — no fire, no
        // skip.
        let observed = stats(8, 60_000_000, 1_000, 10, 250_000);
        let (findings, skips) = evaluate(&Scope::Gateway, day(8), &baseline, &observed);
        assert!(findings.is_empty());
        assert!(skips.is_empty());
    }

    #[test]
    fn zero_baseline_spend_skips_spend_only() {
        let baseline = flat_baseline(0, 1_000, 10, 250_000);
        let observed = stats(8, 60_000_000, 1_000, 10, 250_000);
        let (findings, skips) = evaluate(&Scope::Gateway, day(8), &baseline, &observed);
        assert!(findings.is_empty());
        assert_eq!(skips, vec![SkipReason::ZeroBaselineSpend]);
    }

    #[test]
    fn zero_baseline_latency_skips_latency_only() {
        let baseline = flat_baseline(20_000_000, 1_000, 10, 0);
        let observed = stats(8, 20_000_000, 1_000, 10, 0);
        let (findings, skips) = evaluate(&Scope::Provider, day(8), &baseline, &observed);
        assert!(findings.is_empty());
        assert_eq!(skips, vec![SkipReason::ZeroBaselineLatency]);
    }

    #[test]
    fn zero_baseline_error_rate_skips_error_rate_only() {
        let baseline = flat_baseline(20_000_000, 1_000, 0, 250_000);
        let observed = stats(8, 20_000_000, 1_000, 100, 250_000);
        let (findings, skips) = evaluate(&Scope::Provider, day(8), &baseline, &observed);
        assert!(findings.is_empty());
        assert_eq!(skips, vec![SkipReason::ZeroBaselineErrorRate]);
    }

    #[test]
    fn below_latency_day_request_gate_skips_latency() {
        // 99 < 100 requests on D with an otherwise firing 750ms mean;
        // error_rate's 500-request gate skips too.
        let baseline = flat_baseline(1_000_000, 1_000, 10, 250_000);
        let observed = stats(8, 1_000_000, 99, 1, 74_250);
        let (findings, skips) = evaluate(&Scope::Provider, day(8), &baseline, &observed);
        assert!(findings.is_empty());
        assert_eq!(
            skips,
            vec![
                SkipReason::TooFewRequestsForLatency,
                SkipReason::TooFewRequestsForErrorRate,
            ]
        );
    }

    #[test]
    fn below_error_rate_day_request_gate_skips_error_rate() {
        // 499 < 500 requests on D; latency passes its 100 gate (mean
        // 250ms equals baseline).
        let baseline = flat_baseline(1_000_000, 1_000, 10, 250_000);
        let observed = stats(8, 1_000_000, 499, 249, 124_750);
        let (findings, skips) = evaluate(&Scope::Provider, day(8), &baseline, &observed);
        assert!(findings.is_empty());
        assert_eq!(skips, vec![SkipReason::TooFewRequestsForErrorRate]);
    }

    #[test]
    fn zero_requests_on_evaluated_day_skips_all() {
        let baseline = flat_baseline(20_000_000, 1_000, 10, 250_000);
        let observed = stats(8, 60_000_000, 0, 0, 0);
        let (findings, skips) = evaluate(&Scope::Gateway, day(8), &baseline, &observed);
        assert!(findings.is_empty());
        assert_eq!(skips, vec![SkipReason::NoTrafficOnEvaluatedDay]);
    }

    #[test]
    fn mismatched_observed_day_counts_as_no_traffic() {
        let baseline = flat_baseline(20_000_000, 1_000, 10, 250_000);
        let observed = stats(9, 60_000_000, 2_000, 0, 0);
        let (findings, skips) = evaluate(&Scope::Gateway, day(8), &baseline, &observed);
        assert!(findings.is_empty());
        assert_eq!(skips, vec![SkipReason::NoTrafficOnEvaluatedDay]);
    }

    #[test]
    fn all_four_signals_fire_together() {
        // Dyadic-exact on every axis: $25→$75 (3x, +$50), 1600→16k
        // requests (10x, +14,400), 100/1600→3000/16k errors
        // (0.0625→0.1875, exactly 3x, +12.5pp), 250ms→750ms mean
        // (3x, +500ms).
        let baseline = flat_baseline(25_000_000, 1_600, 100, 400_000);
        let observed = stats(8, 75_000_000, 16_000, 3_000, 12_000_000);
        let (findings, skips) = evaluate(&Scope::Gateway, day(8), &baseline, &observed);
        assert_eq!(
            signals(&findings),
            vec![Signal::Spend, Signal::Traffic, Signal::Latency, Signal::ErrorRate]
        );
        assert!(skips.is_empty());
    }

    #[test]
    fn latency_baseline_is_request_weighted_not_day_averaged() {
        // Every baseline day has a 100ms mean in `flat`; `weighted`
        // swaps 3 days to 9000 requests at 10ms. Day-average would
        // stay 61.43ms for `weighted`; request-weighting yields
        // 670_000/31_000 ≈ 21.61ms.
        let flat = flat_baseline(1_000_000, 1_000, 0, 100_000);
        let mut weighted = flat;
        for d in weighted.iter_mut().take(3) {
            d.requests = 9_000;
            d.duration_ms_sum = 90_000;
        }
        let scope = Scope::Provider;
        let observed = stats(8, 1_000_000, 1_000, 0, 800_000);
        let (f_flat, s_flat) = evaluate(&scope, day(8), &flat, &observed);
        let (f_w, s_w) = evaluate(&scope, day(8), &weighted, &observed);
        assert_eq!(s_flat, vec![SkipReason::ZeroBaselineErrorRate]);
        assert_eq!(s_w, vec![SkipReason::ZeroBaselineErrorRate]);
        let find = |f: &Vec<Finding>| {
            f.iter()
                .find(|x| x.signal == Signal::Latency)
                .cloned()
                .unwrap()
        };
        let a = find(&f_flat);
        let b = find(&f_w);
        assert_eq!(a.baseline_value, 100.0);
        assert!((b.baseline_value - 670_000.0 / 31_000.0).abs() < 1e-9);
        assert!((a.baseline_value - b.baseline_value).abs() > 1.0);
    }

    #[test]
    fn error_rate_baseline_is_request_weighted_not_day_averaged() {
        // 3 days at 1% (10/1000) + 4 days at 10% (10/100):
        // request-weighted 70/3400 ≈ 2.06%; day-average ≈ 6.14%.
        // Observed 10% fires against the weighted baseline (4.86x)
        // but would NOT fire against the day-average (1.63x).
        let mut baseline = flat_baseline(1_000_000, 1_000, 10, 0);
        for d in baseline.iter_mut().skip(3) {
            d.requests = 100;
            d.errors = 10;
        }
        let scope = Scope::Provider;
        let observed = stats(8, 1_000_000, 1_000, 100, 0);
        let (findings, skips) = evaluate(&scope, day(8), &baseline, &observed);
        assert_eq!(skips, vec![SkipReason::ZeroBaselineLatency]);
        let f = findings
            .iter()
            .find(|f| f.signal == Signal::ErrorRate)
            .unwrap();
        assert!((f.baseline_value - 70.0 / 3400.0).abs() < 1e-12);
        assert!((f.observed_value - 0.10).abs() < 1e-12);
    }

    #[test]
    fn error_rate_fires_at_exactly_2pp_delta() {
        // Integer-math boundary: 1% (5/500) → 3% (15/500) is EXACTLY
        // 2pp with exactly a 3x ratio — must fire under integer
        // cross-multiplication (a float path can miss it).
        let baseline = flat_baseline(1_000_000, 500, 5, 250_000);
        let observed = stats(8, 1_000_000, 500, 15, 250_000);
        let (findings, skips) = evaluate(&Scope::Gateway, day(8), &baseline, &observed);
        assert_eq!(signals(&findings), vec![Signal::ErrorRate]);
        assert!(skips.is_empty());
    }

    #[test]
    fn error_rate_just_under_2pp_delta_does_not_fire() {
        // Baseline 0.5% (50/10_000); observed 2% (200/10_000): ratio
        // 4x ≥ 3 but delta 1.5pp < 2pp — must not fire.
        let baseline = flat_baseline(1_000_000, 10_000, 50, 2_500_000);
        let observed = stats(8, 1_000_000, 10_000, 200, 2_500_000);
        let (findings, _) = evaluate(&Scope::Gateway, day(8), &baseline, &observed);
        assert!(
            !findings.iter().any(|f| f.signal == Signal::ErrorRate),
            "1.5pp delta must not fire"
        );
    }

    #[test]
    fn spend_at_49_99usd_delta_does_not_fire() {
        // $10/day baseline, observed $59.99: ratio 5.999x ≥ 3 but
        // delta $49.99 < $50 — must not fire.
        let baseline = flat_baseline(10_000_000, 1_000, 10, 250_000);
        let observed = stats(8, 59_990_000, 1_000, 10, 250_000);
        let (findings, _) = evaluate(&Scope::Gateway, day(8), &baseline, &observed);
        assert!(
            !findings.iter().any(|f| f.signal == Signal::Spend),
            "$49.99 delta must not fire"
        );
    }

    #[test]
    fn traffic_fires_at_exactly_1000_delta() {
        // Baseline mean exactly 100/day (700 total); observed 1100:
        // exactly +1000 over the mean with ratio 11x — fires.
        let baseline = flat_baseline(1_000_000, 100, 1, 10_000);
        let observed = stats(8, 1_000_000, 1_100, 11, 110_000);
        let (findings, _) = evaluate(&Scope::Gateway, day(8), &baseline, &observed);
        assert_eq!(signals(&findings), vec![Signal::Traffic]);
    }

    #[test]
    fn traffic_at_999_delta_does_not_fire() {
        // Baseline mean exactly 100/day (700 total); observed 1099:
        // ratio 10.99x ≥ 10 but +999 < 1000 — must not fire.
        let baseline = flat_baseline(1_000_000, 100, 1, 10_000);
        let observed = stats(8, 1_000_000, 1_099, 11, 109_900);
        let (findings, _) = evaluate(&Scope::Gateway, day(8), &baseline, &observed);
        assert!(
            !findings.iter().any(|f| f.signal == Signal::Traffic),
            "+999 requests must not fire"
        );
    }

    #[test]
    fn latency_at_499ms_delta_does_not_fire() {
        // Baseline mean 100ms (100_000/1_000 × 7 days); observed mean
        // 598.8ms (299_400/500): ratio 5.988x ≥ 3 but delta
        // 498.8ms < 500 — must not fire.
        let baseline = flat_baseline(1_000_000, 1_000, 10, 100_000);
        let observed = stats(8, 1_000_000, 500, 5, 299_400);
        let (findings, _) = evaluate(&Scope::Provider, day(8), &baseline, &observed);
        assert!(
            !findings.iter().any(|f| f.signal == Signal::Latency),
            "498.8ms delta must not fire"
        );
    }

    #[test]
    fn day_request_gates_at_exactly_100_and_500() {
        // Exactly 100 requests on D: latency gate passes (750ms mean
        // on a 250ms baseline fires).
        let baseline = flat_baseline(1_000_000, 1_000, 10, 250_000);
        let observed = stats(8, 1_000_000, 100, 1, 75_000);
        let (findings, skips) = evaluate(&Scope::Provider, day(8), &baseline, &observed);
        assert!(findings.iter().any(|f| f.signal == Signal::Latency));
        assert!(!skips.contains(&SkipReason::TooFewRequestsForLatency));

        // Exactly 500 requests on D: error gate passes (1% → 3%).
        let baseline = flat_baseline(1_000_000, 500, 5, 250_000);
        let observed = stats(8, 1_000_000, 500, 15, 250_000);
        let (findings, skips) = evaluate(&Scope::Gateway, day(8), &baseline, &observed);
        assert!(findings.iter().any(|f| f.signal == Signal::ErrorRate));
        assert!(!skips.contains(&SkipReason::TooFewRequestsForErrorRate));
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    fn any_date() -> impl Strategy<Value = NaiveDate> {
        (0u32..4_000_000u32).prop_map(|n| {
            chrono::NaiveDate::from_ymd_opt(2026, 1, 1)
                .unwrap()
                .checked_add_days(chrono::Days::new(n as u64 % 4000))
                .unwrap()
        })
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 64,
            .. ProptestConfig::default()
        })]

        /// Higher observed spend (same baseline and day traffic)
        /// never yields fewer spend findings; higher observed
        /// requests (same baseline and spend) never yield fewer
        /// traffic findings.
        #[test]
        fn prop_higher_observed_never_fewer_findings(
            base_cost in 0i64..200_000_000,
            extra_cost in 1i64..200_000_000,
            base_req in 1_000i64..10_000,
            day_n in 8u32..=28,
        ) {
            let d = NaiveDate::from_ymd_opt(2026, 11, day_n).unwrap();
            let baseline = [DailyStats {
                day: d - chrono::Days::new(8),
                cost_micro_usd: 10_000_000,
                requests: 1_000,
                errors: 5,
                duration_ms_sum: 100_000,
            }; 7];
            let mk_spend = |cost: i64| DailyStats {
                day: d, cost_micro_usd: cost, requests: 1_000, errors: 0, duration_ms_sum: 0,
            };
            let lo = evaluate(&Scope::Gateway, d, &baseline, &mk_spend(base_cost)).0;
            let hi = evaluate(&Scope::Gateway, d, &baseline, &mk_spend(base_cost + extra_cost)).0;
            prop_assert!(hi.iter().filter(|f| f.signal == Signal::Spend).count()
                >= lo.iter().filter(|f| f.signal == Signal::Spend).count());
            let mk_traffic = |req: i64| DailyStats {
                day: d, cost_micro_usd: 0, requests: req, errors: 0, duration_ms_sum: 0,
            };
            let lo = evaluate(&Scope::Gateway, d, &baseline, &mk_traffic(base_req)).0;
            let hi = evaluate(&Scope::Gateway, d, &baseline, &mk_traffic(base_req * 10 + 1_000)).0;
            prop_assert!(hi.iter().filter(|f| f.signal == Signal::Traffic).count()
                >= lo.iter().filter(|f| f.signal == Signal::Traffic).count());
        }

        /// Higher observed duration sum and error count (fixed
        /// requests) never yield fewer latency / error_rate findings.
        #[test]
        fn prop_higher_observed_value_monotonic_latency_and_error(
            base_dur in 0i64..500_000_000,
            extra_dur in 1i64..500_000_000,
            base_err in 0i64..9_000,
            extra_err in 1i64..9_000,
            day_n in 8u32..=28,
        ) {
            let d = NaiveDate::from_ymd_opt(2026, 11, day_n).unwrap();
            let baseline = [DailyStats {
                day: d - chrono::Days::new(8),
                cost_micro_usd: 10_000_000,
                requests: 20_000,
                errors: 200,
                duration_ms_sum: 20_000_000,
            }; 7];
            let mk = |dur: i64, err: i64| DailyStats {
                day: d, cost_micro_usd: 0, requests: 20_000, errors: err, duration_ms_sum: dur,
            };
            let lo = evaluate(&Scope::Provider, d, &baseline, &mk(base_dur, base_err)).0;
            let hi = evaluate(&Scope::Provider, d, &baseline, &mk(base_dur + extra_dur, base_err + extra_err)).0;
            prop_assert!(hi.iter().filter(|f| f.signal == Signal::Latency).count()
                >= lo.iter().filter(|f| f.signal == Signal::Latency).count());
            prop_assert!(hi.iter().filter(|f| f.signal == Signal::ErrorRate).count()
                >= lo.iter().filter(|f| f.signal == Signal::ErrorRate).count());
        }

        /// Arbitrary (including degenerate: zero, huge, mismatched
        /// dates) aggregates never panic and stay within bounds.
        /// Constraints (review hardening): generated errors never
        /// exceed their day's requests, zero-traffic baseline days
        /// are included, and a mismatched observed day must read as
        /// no-traffic.
        #[test]
        fn prop_no_panic_on_arbitrary_aggregates(
            d in any_date(),
            c0 in 0i64..2_000_000_000, c1 in 0i64..2_000_000_000, c2 in 0i64..2_000_000_000, c3 in 0i64..2_000_000_000, c4 in 0i64..2_000_000_000, c5 in 0i64..2_000_000_000, c6 in 0i64..2_000_000_000,
            r0 in 0i64..1_000_000, r1 in 0i64..1_000_000, r2 in 0i64..1_000_000, r3 in 0i64..1_000_000, r4 in 0i64..1_000_000, r5 in 0i64..1_000_000, r6 in 0i64..1_000_000,
            e0 in 0i64..100_000, e1 in 0i64..100_000, e2 in 0i64..100_000, e3 in 0i64..100_000, e4 in 0i64..100_000, e5 in 0i64..100_000, e6 in 0i64..100_000,
            u0 in 0i64..2_000_000_000, u1 in 0i64..2_000_000_000, u2 in 0i64..2_000_000_000, u3 in 0i64..2_000_000_000, u4 in 0i64..2_000_000_000, u5 in 0i64..2_000_000_000, u6 in 0i64..2_000_000_000,
            o_cost in 0i64..2_000_000_000,
            o_req in 0i64..1_000_000,
            o_err_raw in 0i64..100_000,
            o_dur in 0i64..2_000_000_000,
            mismatched_day in proptest::bool::ANY,
        ) {
            let reqs = [r0, r1, r2, r3, r4, r5, r6];
            // Errors are a subset of requests: cap each day's errors
            // at that day's request count (a real ledger cannot have
            // more failures than finished requests).
            let errs = [e0, e1, e2, e3, e4, e5, e6];
            let costs = [c0, c1, c2, c3, c4, c5, c6];
            let durs = [u0, u1, u2, u3, u4, u5, u6];
            let mut baseline = [DailyStats { day: d, cost_micro_usd: 0, requests: 0, errors: 0, duration_ms_sum: 0 }; 7];
            for i in 0..7 {
                baseline[i] = DailyStats {
                    day: d - chrono::Days::new((7 - i) as u64),
                    cost_micro_usd: costs[i],
                    requests: reqs[i],
                    errors: errs[i].min(reqs[i]),
                    duration_ms_sum: durs[i],
                };
            }
            let o_err = o_err_raw.min(o_req.max(1));
            let observed = DailyStats {
                day: if mismatched_day { d.pred_opt().unwrap() } else { d },
                cost_micro_usd: o_cost,
                requests: o_req,
                errors: o_err,
                duration_ms_sum: o_dur,
            };
            let (findings, skips) = evaluate(&Scope::ModelGroup, d, &baseline, &observed);
            if mismatched_day || o_req == 0 {
                // Mismatched/empty observed day: no traffic, so no
                // findings may fire at all.
                prop_assert!(findings.is_empty());
                prop_assert!(skips.contains(&SkipReason::NoTrafficOnEvaluatedDay));
            }
            prop_assert!(findings.len() <= 4);
            prop_assert!(skips.len() <= 7);
            for f in &findings {
                prop_assert!(f.observed_value.is_finite());
                prop_assert!(f.baseline_value.is_finite());
            }
        }

        /// Purity: identical inputs yield identical outputs.
        #[test]
        fn prop_evaluate_is_pure(
            d in any_date(),
            seed_cost in 0u64..2_000_000_000,
            o_req in 0i64..1_000_000,
            o_err in 0i64..100_000,
        ) {
            let baseline = [DailyStats {
                day: d,
                cost_micro_usd: (seed_cost % 100_000_000) as i64,
                requests: 1_000,
                errors: (seed_cost % 97) as i64,
                duration_ms_sum: (seed_cost % 500_000) as i64,
            }; 7];
            let observed = DailyStats {
                day: d,
                cost_micro_usd: (seed_cost % 90_000_000) as i64,
                requests: o_req,
                errors: o_err,
                duration_ms_sum: (seed_cost % 800_000) as i64,
            };
            let a = evaluate(&Scope::Gateway, d, &baseline, &observed);
            let b = evaluate(&Scope::Gateway, d, &baseline, &observed);
            prop_assert_eq!(a, b);
        }
    }
}
