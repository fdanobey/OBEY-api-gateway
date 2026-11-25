//! Pure evaluation core for usage-anomaly alerts (spec:
//! usage-anomaly-alerts, Component 1).
//!
//! Decides, from caller-supplied per-key daily and behavioral
//! aggregates, whether a (key, signal, day) finding fires for one
//! evaluated UTC day against the 7 days preceding it. The module
//! is deliberately pure â€” no I/O, no async, no clocks, no
//! config or metrics imports: identical inputs always yield
//! identical findings and skips (the `performance_alerts/state.rs`
//! discipline). Baselines are window sums cross-multiplied over
//! the 7-day denominator, never float divisions.
//!
//! Fixed thresholds and minimum-data gates (design table):
//!
//! | Signal | Ratio | Absolute leg | Skip when |
//! |--------------|--------------------|------------------|----------------------------------------------|
//! | spend | >=3x daily baseline | >=$50 | baseline spend = 0, or <4 traffic days |
//! | traffic | >=10x daily baseline | >=1000 req | <4 traffic days |
//! | tokens | >=10x daily baseline | >=1M tokens | <4 traffic days |
//! | new_activity | zero baseline only | >=25 req | any baseline traffic |
//! | odd_hours | >=50% in sub-2% buckets | >=200 req | <4 traffic days, or no baseline requests |
//! | new_model | absent from baseline | >=10 req | <4 traffic days |
//! | probes | >=5x daily baseline | >=50 failures | <4 traffic days (absolute leg tolerates) |
//!
//! Baseline day counts as "with traffic" if requests > 0. A
//! baseline day-count under 4 of 7, or zero baseline traffic,
//! skips the ratio-defined signals; `new_activity` needs the
//! zero baseline and `probes`' absolute leg tolerates one.
//!
//! _Requirements: 1.1â€“1.10, 2.1â€“2.5, 6.1_

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

/// Minimum baseline days with traffic before a ratio-defined
/// signal is trusted (design "Skip when": traffic on fewer than
/// 4 of the 7 baseline days).
pub const MIN_BASELINE_TRAFFIC_DAYS: usize = 4;

/// Spend fires at 3x or more of the daily baseline with at
/// least $50 absolute drift. Integer, cross-multiplied
/// against the 7-day baseline sum in `evaluate` (exact
/// thresholds â€” no float drift).
pub const SPEND_RATIO: i128 = 3;
/// Minimum absolute spend drift: $50, in micro-USD.
pub const SPEND_MIN_DELTA_MICRO_USD: i128 = 50_000_000;

/// Traffic fires at 10x or more of the daily baseline with at
/// least 1000 absolute requests (integer
/// cross-multiplication).
pub const TRAFFIC_RATIO: i128 = 10;
/// Minimum absolute traffic drift: 1000 requests.
pub const TRAFFIC_MIN_DELTA: i128 = 1_000;

/// Tokens fires at 10x or more of the daily baseline with at
/// least 1,000,000 tokens (input + output).
pub const TOKENS_RATIO: i128 = 10;
/// Minimum absolute token drift: 1,000,000 tokens.
pub const TOKENS_MIN_DELTA: i128 = 1_000_000;

/// Odd-hours fires when 50% or more of the evaluated day's
/// requests land in hour buckets whose baseline share is under
/// 2%, with at least 200 requests on the day.
pub const ODD_HOURS_SHARE_NUM: i128 = 1;
/// Denominator of the odd-hours day-share threshold (50%).
pub const ODD_HOURS_SHARE_DEN: i128 = 2;
/// A baseline bucket counts as quiet when its request share is
/// under 1/50 (2%).
pub const QUIET_BUCKET_SHARE_NUM: i128 = 1;
/// Denominator of the quiet-bucket share threshold (2%).
pub const QUIET_BUCKET_SHARE_DEN: i128 = 50;
/// Minimum requests on the evaluated day before odd_hours is
/// trusted.
pub const ODD_HOURS_MIN_DAY_REQUESTS: i64 = 200;

/// New-model fires at 10 or more requests to a model absent
/// from the key's baseline model set.
pub const NEW_MODEL_MIN_REQUESTS: i64 = 10;

/// Probes fires at 50 or more validation-class failures AND
/// 5x or more of the baseline daily count (the absolute leg
/// tolerates a zero baseline).
pub const PROBES_RATIO: i128 = 5;
pub const PROBES_MIN_DELTA: i128 = 50;

/// New-activity fires at 25 or more requests on the evaluated
/// day against zero baseline traffic.
pub const NEW_ACTIVITY_MIN_REQUESTS: i64 = 25;

/// Baseline window length (days): the cross-multiplication
/// denominator that replaces the per-day average in fire
/// decisions.
const BASELINE_WINDOW_DAYS: i128 = 7;

/// Baseline window length as a row count (len checks and
/// test helpers); kept in lockstep with
/// [`BASELINE_WINDOW_DAYS`].
const BASELINE_WINDOW_LEN: usize = 7;

/// Micro-USD per USD for the spend DISPLAY values only â€” fire
/// and no-fire decisions use the integer micro-USD sums
/// directly.
const MICRO_USD_PER_USD: f64 = 1_000_000.0;

/// Number of 4-hour UTC buckets in a day.
const HOUR_BUCKETS: usize = 6;

/// One day of per-key ledger aggregates (design Data Models;
/// `validation_failures` arrives precomputed by the query layer
/// as COUNT(status_code IN (400,401,403) AND output_tokens =
/// 0) â€” 429 excluded).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeyDailyStats {
    /// UTC day the row buckets.
    pub day: NaiveDate,
    /// Î£ cost in micro-USD for the day.
    pub cost_micro_usd: i64,
    /// Î£ finished requests for the day.
    pub requests: i64,
    /// Î£ validation-class failures for the day.
    pub validation_failures: i64,
    /// Î£ input tokens for the day.
    pub input_tokens: i64,
    /// Î£ output tokens for the day.
    pub output_tokens: i64,
}

impl KeyDailyStats {
    /// Total token throughput (input + output) for the day.
    fn total_tokens(&self) -> i128 {
        self.input_tokens as i128 + self.output_tokens as i128
    }
}

/// Request counts for one key across the six 4-hour UTC
/// buckets (index 0 = 00:00â€“03:59, â€¦, index 5 = 20:00â€“23:59).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeyHourHistogram {
    /// Per-4h-bucket request counts.
    pub buckets: [i64; HOUR_BUCKETS],
}

/// Request count for one (key, model) pair on the evaluated
/// day; the baseline model set is built from baseline-day rows.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeyModelUsage {
    /// Requested model name.
    pub model: String,
    /// Î£ requests for the model.
    pub requests: i64,
}

/// Stable identity of one virtual key: the id is the dedup
/// identity; the display name is cosmetic (never the `vk_`
/// secret).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct KeyRef {
    /// Stable key id.
    pub key_id: String,
    /// Display name, when known.
    pub name: Option<String>,
}

/// Usage-anomaly signal under evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Signal {
    Spend,
    Traffic,
    Tokens,
    NewActivity,
    OddHours,
    NewModel,
    Probes,
}

impl Signal {
    /// Serde-friendly discriminant for payloads.
    pub fn as_str(&self) -> &'static str {
        match self {
            Signal::Spend => "spend",
            Signal::Traffic => "traffic",
            Signal::Tokens => "tokens",
            Signal::NewActivity => "new_activity",
            Signal::OddHours => "odd_hours",
            Signal::NewModel => "new_model",
            Signal::Probes => "probes",
        }
    }
}

/// Unit accompanying a [`Finding`]'s values.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Unit {
    Usd,
    Requests,
    Tokens,
    Ratio,
}

/// A fired (key, signal, day) comparison: observed vs baseline
/// with the fixed condition that fired.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Finding {
    /// Key the finding fired for.
    pub key: KeyRef,
    /// Signal that fired.
    pub signal: Signal,
    /// Evaluated UTC day.
    pub day: NaiveDate,
    /// Observed day's value for the signal.
    pub observed_value: f64,
    /// Baseline (7-day) value for the signal.
    pub baseline_value: f64,
    /// Unit both values share.
    pub unit: Unit,
    /// Signal-specific detail (bucket share, model name,
    /// validation-failure count).
    pub detail: String,
    /// Human-readable fixed condition that fired.
    pub threshold: &'static str,
}

/// A minimum-data gate that suppressed one signal's check.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SkipReason {
    /// Traffic on fewer than 4 of the 7 baseline days
    /// (ratio-defined signals).
    InsufficientBaselineDays,
    /// Baseline traffic is zero for the ratio-defined signals
    /// (new_activity fires instead).
    ZeroBaseline,
    /// Zero requests on the evaluated day.
    NoTrafficOnEvaluatedDay,
    /// No behavioral aggregates supplied (odd_hours /
    /// new_model undefined).
    NoBehavioralData,
    /// Baseline rows are malformed for the evaluated day:
    /// wrong length, duplicate days, a day outside the
    /// 7-day window, or the evaluated day itself.
    MalformedBaseline,
}

/// One recorded skip: the key, the day, and the reason one
/// signal's check was suppressed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Skip {
    /// Key the skip was recorded for.
    pub key: KeyRef,
    /// Evaluated UTC day.
    pub day: NaiveDate,
    /// Why the check was suppressed.
    pub reason: SkipReason,
}

/// Inputs for one key's evaluation: the evaluated day's
/// aggregates, the 7 preceding days, and the optional
/// behavioral aggregates (hour histogram + model usage for
/// observed day and baseline model set).
#[derive(Debug, Clone, PartialEq)]
pub struct KeyEvaluation {
    /// Stable key identity.
    pub key: KeyRef,
    /// The evaluated day's daily aggregates.
    pub observed: KeyDailyStats,
    /// The 7 baseline days preceding the evaluated day.
    pub baseline: Vec<KeyDailyStats>,
    /// Observed-day hour-bucket histogram.
    pub hour_histogram: Option<KeyHourHistogram>,
    pub baseline_hour_histogram: Option<KeyHourHistogram>,
    /// Observed-day per-model request counts.
    pub observed_models: Vec<KeyModelUsage>,
    /// Model names with requests on any baseline day.
    pub baseline_models: Option<Vec<String>>,
}

/// Evaluate one key's day against the 7 days preceding it:
/// findings for every fired signal plus every minimum-data
/// skip (Req 2.5). Pure â€” same inputs always yield the same
/// output.
pub fn evaluate(eval: &KeyEvaluation, day: NaiveDate) -> (Vec<Finding>, Vec<Skip>) {
    let mut findings = Vec::new();
    let mut skips = Vec::new();
    let key = &eval.key;
    let base = &eval.baseline;
    let window_start = day.checked_sub_days(chrono::Days::new(BASELINE_WINDOW_LEN as u64));
    let mut has_seven = base.len() == BASELINE_WINDOW_LEN && window_start.is_some();
    if let Some(start) = window_start {
        let mut seen = [false; BASELINE_WINDOW_LEN];
        for b in base {
            match b.day.signed_duration_since(start).num_days() {
                index if index >= 0 && (index as usize) < BASELINE_WINDOW_LEN => {
                    let index = index as usize;
                    if seen[index] {
                        has_seven = false;
                        break;
                    }
                    seen[index] = true;
                }
                _ => {
                    has_seven = false;
                    break;
                }
            }
        }
    }
    if !has_seven {
        skips.push(Skip {
            key: key.clone(),
            day,
            reason: SkipReason::MalformedBaseline,
        });
        return (findings, skips);
    }
    let traffic_days = base.iter().filter(|b| b.requests > 0).count();
    let enough_days = traffic_days >= MIN_BASELINE_TRAFFIC_DAYS;
    let baseline_traffic_total: i128 = base.iter().map(|b| b.requests as i128).sum();
    let zero_baseline = baseline_traffic_total <= 0;
    let day_requests = if eval.observed.day == day {
        eval.observed.requests
    } else {
        0
    };

    if day_requests <= 0 {
        skips.push(Skip {
            key: key.clone(),
            day,
            reason: SkipReason::NoTrafficOnEvaluatedDay,
        });
        if !enough_days {
            skips.push(Skip {
                key: key.clone(),
                day,
                reason: SkipReason::InsufficientBaselineDays,
            });
        }
        return (findings, skips);
    }
    if !enough_days {
        skips.push(Skip {
            key: key.clone(),
            day,
            reason: SkipReason::InsufficientBaselineDays,
        });
    }
    if zero_baseline {
        skips.push(Skip {
            key: key.clone(),
            day,
            reason: SkipReason::ZeroBaseline,
        });
    }

    // Integer decision math (drift-free thresholds): every fire
    // comparison is an i128 cross-multiplication over the raw
    // i64 ledger sums, never a float division. All inputs are
    // non-negative ledger aggregates; the practical bounds
    // (â‰¤ ~9.2e18 per field) keep every product far below the
    // i128 limit. The f64 values on the findings are
    // DISPLAY-ONLY, computed with safe division.
    let baseline_cost_micro: i128 = base.iter().map(|b| b.cost_micro_usd as i128).sum();
    let observed_cost_micro: i128 = eval.observed.cost_micro_usd as i128;
    if enough_days && !zero_baseline {
        if baseline_cost_micro <= 0 {
            // spend needs a nonzero baseline sum
        } else if observed_cost_micro * BASELINE_WINDOW_DAYS >= SPEND_RATIO * baseline_cost_micro
            && observed_cost_micro * BASELINE_WINDOW_DAYS - baseline_cost_micro
                >= SPEND_MIN_DELTA_MICRO_USD * BASELINE_WINDOW_DAYS
        {
            findings.push(Finding {
                key: key.clone(),
                signal: Signal::Spend,
                day,
                observed_value: observed_cost_micro as f64 / MICRO_USD_PER_USD,
                baseline_value: baseline_cost_micro as f64
                    / MICRO_USD_PER_USD
                    / BASELINE_WINDOW_DAYS as f64,
                unit: Unit::Usd,
                detail: String::new(),
                threshold: ">=3x baseline and >=$50 over",
            });
        }
    }

    let observed_requests: i128 = day_requests as i128;
    if enough_days && !zero_baseline {
        if observed_requests * BASELINE_WINDOW_DAYS >= TRAFFIC_RATIO * baseline_traffic_total
            && observed_requests * BASELINE_WINDOW_DAYS - baseline_traffic_total
                >= TRAFFIC_MIN_DELTA * BASELINE_WINDOW_DAYS
        {
            findings.push(Finding {
                key: key.clone(),
                signal: Signal::Traffic,
                day,
                observed_value: observed_requests as f64,
                baseline_value: baseline_traffic_total as f64 / BASELINE_WINDOW_DAYS as f64,
            unit: Unit::Requests,
            detail: String::new(),
            threshold: ">=10x baseline and >=1000 over",
            });
        }
    }

    let baseline_tokens: i128 = base.iter().map(|b| b.total_tokens()).sum();
    let observed_tokens: i128 = eval.observed.total_tokens();
    if enough_days && !zero_baseline {
        if observed_tokens * BASELINE_WINDOW_DAYS >= TOKENS_RATIO * baseline_tokens
            && observed_tokens * BASELINE_WINDOW_DAYS - baseline_tokens
                >= TOKENS_MIN_DELTA * BASELINE_WINDOW_DAYS
        {
            findings.push(Finding {
                key: key.clone(),
                signal: Signal::Tokens,
                day,
                observed_value: observed_tokens as f64,
                baseline_value: baseline_tokens as f64 / BASELINE_WINDOW_DAYS as f64,
                unit: Unit::Tokens,
                detail: String::new(),
                threshold: ">=10x baseline and >=1,000,000 over",
            });
        }
    }

    if zero_baseline {
        if day_requests >= NEW_ACTIVITY_MIN_REQUESTS {
            findings.push(Finding {
                key: key.clone(),
                signal: Signal::NewActivity,
                day,
                observed_value: observed_requests as f64,
                baseline_value: 0.0,
                unit: Unit::Requests,
                detail: String::new(),
                threshold: ">=25 requests with zero baseline traffic",
            });
        }
    }

    if let Some(hist) = eval.hour_histogram.as_ref() {
        let hist_total: i128 = hist.buckets.iter().map(|b| *b as i128).sum();
        let baseline_hist = eval.baseline_hour_histogram.as_ref();
        if hist_total != day_requests as i128
            || baseline_traffic_total <= 0
            || baseline_hist.is_none()
        {
            // histogram disagrees with the daily row (or no
            // baseline histogram to classify quiet buckets
            // against): treat as absent so the signal stays
            // dormant, not wrong.
        } else if !enough_days {
            // insufficient-baseline skip already recorded
        } else if day_requests < ODD_HOURS_MIN_DAY_REQUESTS {
            // below the absolute leg: no finding, no skip
        } else {
            let baseline_hist = baseline_hist.unwrap();
            let mut odd_requests: i128 = 0;
            let mut quiet_buckets = 0;
            for (bucket, baseline_bucket) in hist.buckets.iter().zip(baseline_hist.buckets.iter()) {
                let bucket_requests: i128 = *bucket as i128;
                let baseline_bucket_requests: i128 = *baseline_bucket as i128;
                if bucket_requests > 0
                    && baseline_bucket_requests * QUIET_BUCKET_SHARE_DEN
                        < QUIET_BUCKET_SHARE_NUM * baseline_traffic_total
                {
                    odd_requests += bucket_requests;
                    quiet_buckets += 1;
                }
            }
            if odd_requests * ODD_HOURS_SHARE_DEN >= ODD_HOURS_SHARE_NUM * hist_total {
                let baseline_odd_requests: i128 = baseline_hist
                    .buckets
                    .iter()
                    .zip(hist.buckets.iter())
                    .filter(|(baseline_bucket, bucket)| {
                        **bucket > 0
                            && **baseline_bucket as i128 * QUIET_BUCKET_SHARE_DEN
                                < QUIET_BUCKET_SHARE_NUM * baseline_traffic_total
                    })
                    .map(|(baseline_bucket, _)| *baseline_bucket as i128)
                    .sum::<i128>();
                findings.push(Finding {
                    key: key.clone(),
                    signal: Signal::OddHours,
                    day,
                    observed_value: odd_requests as f64 / hist_total as f64,
                    baseline_value: baseline_odd_requests as f64 / hist_total as f64,
                    unit: Unit::Ratio,
                    detail: format!(
                        "{odd_requests} of {hist_total} requests in {quiet_buckets} baseline-quiet buckets"
                    ),
                    threshold: ">=50% of day requests in baseline-under-2% buckets and >=200 requests",
                });
            }
    }
} else if enough_days
    && baseline_traffic_total > 0
    && day_requests >= ODD_HOURS_MIN_DAY_REQUESTS
{
        skips.push(Skip {
            key: key.clone(),
            day,
            reason: SkipReason::NoBehavioralData,
        });
    }

    match eval.baseline_models.as_ref() {
        Some(baseline_models) => {
            if !enough_days {
                // insufficient-baseline skip already recorded
            } else {
                // Per-model gate (>=10 requests to EACH
                // baseline-absent model), but ONE finding per key
                // and day: the dedup identity is
                // (key id, signal, day), so per-model findings
                // would collapse into one ledger row anyway —
                // collapse in the core and carry the BUSIEST
                // qualifying model (ties broken
                // lexicographically for determinism).
                let mut busiest: Option<&KeyModelUsage> = None;
                for usage in eval.observed_models.iter() {
                    if usage.requests >= NEW_MODEL_MIN_REQUESTS
                        && !baseline_models.iter().any(|m| m == &usage.model)
                        && busiest
                            .map(|best| {
                                usage.requests > best.requests
                                    || (usage.requests == best.requests
                                        && usage.model < best.model)
                            })
                            .unwrap_or(true)
                    {
                        busiest = Some(usage);
                    }
                }
                if let Some(usage) = busiest {
                    findings.push(Finding {
                        key: key.clone(),
                        signal: Signal::NewModel,
                        day,
                        observed_value: usage.requests as f64,
                        baseline_value: 0.0,
                        unit: Unit::Requests,
                        detail: usage.model.clone(),
                        threshold: ">=10 requests to a model absent from the key's baseline",
                    });
                }
            }
        }
        None => {
            if enough_days {
                skips.push(Skip {
                    key: key.clone(),
                    day,
                    reason: SkipReason::NoBehavioralData,
                });
            }
        }
    }

    let baseline_failures: i128 = base.iter().map(|b| b.validation_failures as i128).sum();
    let observed_failures: i128 = eval.observed.validation_failures as i128;
    let absolute_fires = observed_failures >= PROBES_MIN_DELTA;
    let ratio_fires = !zero_baseline
        && enough_days
        && baseline_failures > 0
        && observed_failures * BASELINE_WINDOW_DAYS >= PROBES_RATIO * baseline_failures;
    if absolute_fires && ratio_fires {
        findings.push(Finding {
            key: key.clone(),
            signal: Signal::Probes,
            day,
            observed_value: observed_failures as f64,
            baseline_value: baseline_failures as f64 / BASELINE_WINDOW_DAYS as f64,
            unit: Unit::Requests,
            detail: format!("{observed_failures} validation-class failures"),
            threshold: ">=50 validation failures and >=5x baseline",
        });
    } else if absolute_fires && zero_baseline {
        findings.push(Finding {
            key: key.clone(),
            signal: Signal::Probes,
            day,
            observed_value: observed_failures as f64,
            baseline_value: 0.0,
            unit: Unit::Requests,
            detail: format!("{observed_failures} validation-class failures"),
            threshold: ">=50 validation failures and >=5x baseline",
        });
    }

    (findings, skips)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn day(n: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(2026, 11, n).unwrap()
    }

    fn stats(n: u32, requests: i64) -> KeyDailyStats {
        KeyDailyStats {
            day: day(n),
            cost_micro_usd: 0,
            requests,
            validation_failures: 0,
            input_tokens: 0,
            output_tokens: 0,
        }
    }

    fn key() -> KeyRef {
        KeyRef {
            key_id: "k1".to_string(),
            name: Some("prod".to_string()),
        }
    }

fn flat_baseline(requests: i64) -> Vec<KeyDailyStats> {
    (1..=BASELINE_WINDOW_LEN as u32).map(|n| stats(n, requests)).collect()
}

    fn eval(baseline: Vec<KeyDailyStats>, observed: KeyDailyStats) -> KeyEvaluation {
        KeyEvaluation {
            key: key(),
            observed,
            baseline,
            hour_histogram: None,
            baseline_hour_histogram: None,
            observed_models: Vec::new(),
            baseline_models: None,
        }
    }

    fn signals(findings: &[Finding]) -> Vec<Signal> {
        findings.iter().map(|f| f.signal).collect()
    }

    fn reasons(skips: &[Skip]) -> Vec<SkipReason> {
        skips.iter().map(|s| s.reason).collect()
    }

    fn one_finding(findings: &[Finding], signal: Signal) -> &Finding {
        findings.iter().find(|f| f.signal == signal).unwrap()
    }

    /// The eval() helper leaves both behavioral aggregates
    /// absent, so otherwise-eligible days record only the two
    /// NoBehavioralData skips (hour histogram + model set).
    fn assert_no_data_skips_only(skips: &[Skip]) {
        assert!(skips
            .iter()
            .all(|s| s.reason == SkipReason::NoBehavioralData));
    }

    // ---- spend ----

    #[test]
    fn spend_fires_exactly_at_thresholds() {
        // $25/day baseline => observed $75.000001 = just over 3x
        // with just over $50 over: fires.
        let baseline: Vec<KeyDailyStats> = (1..=BASELINE_WINDOW_LEN as u32)
            .map(|n| KeyDailyStats { cost_micro_usd: 25_000_000, ..stats(n, 1_000) })
            .collect();
        let observed = KeyDailyStats { cost_micro_usd: 75_000_001, ..stats(8, 1_000) };
        let (findings, skips) = evaluate(&eval(baseline, observed), day(8));
        assert_eq!(signals(&findings), vec![Signal::Spend]);
        assert_no_data_skips_only(&skips);
        let f = one_finding(&findings, Signal::Spend);
        assert_eq!(f.unit, Unit::Usd);
        assert_eq!(f.observed_value, 75.000001);
        assert_eq!(f.baseline_value, 25.0);
    }

#[test]
fn spend_at_exact_ratio_and_delta_fires() {
    // Exactly 3x and exactly $50 over: both legs are
    // inclusive (>=, >=$50) so it fires.
    let baseline: Vec<KeyDailyStats> = (1..=BASELINE_WINDOW_LEN as u32)
        .map(|n| KeyDailyStats { cost_micro_usd: 25_000_000, ..stats(n, 1_000) })
        .collect();
    let observed = KeyDailyStats { cost_micro_usd: 75_000_000, ..stats(8, 1_000) };
    let (findings, skips) = evaluate(&eval(baseline, observed), day(8));
    assert_eq!(signals(&findings), vec![Signal::Spend]);
    assert_no_data_skips_only(&skips);
}

#[test]
fn spend_just_below_ratio_or_delta_does_not_fire() {
    // $25/day baseline: $74.999999 fails the $50-delta leg at
    // the nearest representable point below the paired
    // threshold, and a strictly-under-3x spend fails the
    // ratio leg even with a large delta.
    let baseline: Vec<KeyDailyStats> = (1..=BASELINE_WINDOW_LEN as u32)
        .map(|n| KeyDailyStats { cost_micro_usd: 25_000_000, ..stats(n, 1_000) })
        .collect();
    let observed = KeyDailyStats { cost_micro_usd: 74_999_999, ..stats(8, 1_000) };
    let (findings, skips) = evaluate(&eval(baseline.clone(), observed), day(8));
    assert!(findings.is_empty());
    assert_no_data_skips_only(&skips);
    // $20/day => 3x is $60; $59.999999 is just under the
    // ratio leg ($39.999999 over passes the delta leg).
    let baseline: Vec<KeyDailyStats> = (1..=BASELINE_WINDOW_LEN as u32)
        .map(|n| KeyDailyStats { cost_micro_usd: 20_000_000, ..stats(n, 1_000) })
        .collect();
    let observed = KeyDailyStats { cost_micro_usd: 59_999_999, ..stats(8, 1_000) };
    let (findings, skips) = evaluate(&eval(baseline, observed), day(8));
    assert!(findings.is_empty());
    assert_no_data_skips_only(&skips);
}

    #[test]
    fn spend_below_delta_does_not_fire() {
        // 5x ratio but only $40 over: absolute leg fails.
        let baseline: Vec<KeyDailyStats> = (1..=BASELINE_WINDOW_LEN as u32)
            .map(|n| KeyDailyStats { cost_micro_usd: 10_000_000, ..stats(n, 1_000) })
            .collect();
        let observed = KeyDailyStats { cost_micro_usd: 50_000_000, ..stats(8, 1_000) };
        let (findings, skips) = evaluate(&eval(baseline, observed), day(8));
        assert!(findings.is_empty());
        assert_no_data_skips_only(&skips);
    }

#[test]
fn spend_at_49_999999_delta_does_not_fire() {
    // $25/day baseline, observed $74.999999: ratio
    // 2.99999996x fails AND $49.999999 over fails —
    // nearest representable just-below-delta point.
    let baseline: Vec<KeyDailyStats> = (1..=BASELINE_WINDOW_LEN as u32)
        .map(|n| KeyDailyStats { cost_micro_usd: 25_000_000, ..stats(n, 1_000) })
        .collect();
    let observed = KeyDailyStats { cost_micro_usd: 74_999_999, ..stats(8, 1_000) };
    let (findings, skips) = evaluate(&eval(baseline, observed), day(8));
    assert!(findings.is_empty());
    assert_no_data_skips_only(&skips);
}

// ---- traffic ----

#[test]
fn traffic_fires_exactly_at_thresholds() {
    // 200/day => 2000 observed = exactly 10x (inclusive) with
    // exactly the 1800 >= 1000 delta leg: fires.
    let baseline = flat_baseline(200);
    let observed = stats(8, 2_000);
    let (findings, skips) = evaluate(&eval(baseline, observed), day(8));
    assert_eq!(signals(&findings), vec![Signal::Traffic]);
    assert_no_data_skips_only(&skips);
    let f = one_finding(&findings, Signal::Traffic);
    assert_eq!(f.observed_value, 2_000.0);
    assert_eq!(f.baseline_value, 200.0);
    assert_eq!(f.unit, Unit::Requests);
}

#[test]
fn traffic_at_999_delta_does_not_fire() {
    // 100/day => 1099 observed: ratio passes but the delta
    // is 999 < 1000.
    let baseline = flat_baseline(100);
    let observed = stats(8, 1_099);
    let (findings, skips) = evaluate(&eval(baseline, observed), day(8));
    assert!(findings.is_empty());
    assert_no_data_skips_only(&skips);
}

#[test]
fn traffic_just_below_10x_stays_quiet() {
    // 200/day => 1999 observed = 9.995x (just under the
    // inclusive 10x ratio) with 1799 over: ratio leg fails.
    let baseline = flat_baseline(200);
    let observed = stats(8, 1_999);
    let (findings, skips) = evaluate(&eval(baseline, observed), day(8));
    assert!(findings.is_empty());
    assert_no_data_skips_only(&skips);
}

    // ---- tokens ----

#[test]
fn tokens_fires_exactly_at_thresholds() {
    // 100k tokens/day => 1_100_000 observed: exactly 11x
    // (>=10x) with exactly 1M over: both legs fire at the
    // boundary.
    let baseline: Vec<KeyDailyStats> = (1..=BASELINE_WINDOW_LEN as u32)
        .map(|n| KeyDailyStats { input_tokens: 100_000, ..stats(n, 100) })
        .collect();
    let observed = KeyDailyStats { input_tokens: 1_100_000, ..stats(8, 100) };
    let (findings, skips) = evaluate(&eval(baseline, observed), day(8));
    assert_eq!(signals(&findings), vec![Signal::Tokens]);
    assert_no_data_skips_only(&skips);
    let f = one_finding(&findings, Signal::Tokens);
    assert_eq!(f.observed_value, 1_100_000.0);
    assert_eq!(f.baseline_value, 100_000.0);
    assert_eq!(f.unit, Unit::Tokens);
}

#[test]
fn tokens_at_exact_delta_fires_and_just_below_does_not() {
    // 100k/day => 1_100_000 observed = 11x with exactly the
    // inclusive 1M delta: fires. 1_099_999 does not.
    let baseline: Vec<KeyDailyStats> = (1..=BASELINE_WINDOW_LEN as u32)
        .map(|n| KeyDailyStats { input_tokens: 100_000, ..stats(n, 100) })
        .collect();
    let observed = KeyDailyStats { input_tokens: 1_100_000, ..stats(8, 100) };
    let (findings, _) = evaluate(&eval(baseline.clone(), observed), day(8));
    assert_eq!(signals(&findings), vec![Signal::Tokens]);
    let observed = KeyDailyStats { input_tokens: 1_099_999, ..stats(8, 100) };
    let (findings, skips) = evaluate(&eval(baseline, observed), day(8));
    assert!(findings.is_empty());
    assert_no_data_skips_only(&skips);
}

#[test]
fn tokens_exact_10x_fires() {
    // 1M/day => 10M observed: exactly 10x (>=10x inclusive)
    // with 9M over: both legs pass at the boundary.
    let baseline: Vec<KeyDailyStats> = (1..=BASELINE_WINDOW_LEN as u32)
        .map(|n| KeyDailyStats { input_tokens: 1_000_000, ..stats(n, 100) })
        .collect();
    let observed = KeyDailyStats { input_tokens: 10_000_000, ..stats(8, 100) };
    let (findings, skips) = evaluate(&eval(baseline, observed), day(8));
    assert_eq!(signals(&findings), vec![Signal::Tokens]);
    assert_no_data_skips_only(&skips);
}

#[test]
fn tokens_just_below_10x_stays_quiet() {
    // 1M/day => 9_999_999 observed: just under the inclusive
    // 10x ratio even though the delta passes.
    let baseline: Vec<KeyDailyStats> = (1..=BASELINE_WINDOW_LEN as u32)
        .map(|n| KeyDailyStats { input_tokens: 1_000_000, ..stats(n, 100) })
        .collect();
    let observed = KeyDailyStats { input_tokens: 9_999_999, ..stats(8, 100) };
    let (findings, skips) = evaluate(&eval(baseline, observed), day(8));
    assert!(findings.is_empty());
    assert_no_data_skips_only(&skips);
}

    #[test]
    fn tokens_counts_input_plus_output() {
        // total_tokens = input + output on both sides.
        let baseline: Vec<KeyDailyStats> = (1..=BASELINE_WINDOW_LEN as u32)
            .map(|n| KeyDailyStats {
                input_tokens: 50_000,
                output_tokens: 50_000,
                ..stats(n, 100)
            })
            .collect();
        let observed = KeyDailyStats {
            input_tokens: 550_001,
            output_tokens: 550_000,
            ..stats(8, 100)
        };
        let (findings, _) = evaluate(&eval(baseline, observed), day(8));
        assert_eq!(signals(&findings), vec![Signal::Tokens]);
    }

    // ---- new_activity ----

    #[test]
    fn new_activity_fires_at_25_on_zero_baseline() {
        let baseline = flat_baseline(0);
        let observed = stats(8, 25);
        let (findings, skips) = evaluate(&eval(baseline, observed), day(8));
        assert_eq!(signals(&findings), vec![Signal::NewActivity]);
        // zero baseline + absent behavioral data: ZeroBaseline
        // plus NoBehavioralData for the model set (day requests
        // 25 < 200 so the hour histogram records nothing).
        assert!(reasons(&skips).contains(&SkipReason::ZeroBaseline));
        let f = one_finding(&findings, Signal::NewActivity);
        assert_eq!(f.observed_value, 25.0);
        assert_eq!(f.baseline_value, 0.0);
    }

    #[test]
    fn new_activity_at_24_does_not_fire() {
        let baseline = flat_baseline(0);
        let observed = stats(8, 24);
        let (findings, skips) = evaluate(&eval(baseline, observed), day(8));
        assert!(findings.is_empty());
        assert!(reasons(&skips).contains(&SkipReason::ZeroBaseline));
    }

    #[test]
    fn new_activity_never_fires_with_any_baseline_traffic() {
        // One baseline day with a single request makes the
        // baseline nonzero: new_activity is impossible and no
        // ZeroBaseline skip is recorded.
        let mut baseline = flat_baseline(0);
        baseline[0].requests = 1;
        let observed = stats(8, 10_000);
        let (findings, skips) = evaluate(&eval(baseline, observed), day(8));
        assert!(!findings.iter().any(|f| f.signal == Signal::NewActivity));
        assert!(!reasons(&skips).contains(&SkipReason::ZeroBaseline));
    }

    // ---- odd_hours ----

    fn odd_eval(
        baseline: &[KeyDailyStats],
        observed: KeyDailyStats,
        hist: [i64; 6],
        baseline_hist: [i64; 6],
    ) -> KeyEvaluation {
        KeyEvaluation {
            hour_histogram: Some(KeyHourHistogram { buckets: hist }),
            baseline_hour_histogram: Some(KeyHourHistogram { buckets: baseline_hist }),
            ..eval(baseline.to_vec(), observed)
        }
    }

    #[test]
    fn odd_hours_fires_when_majority_in_baseline_quiet_buckets() {
        // Baseline 400/day all in bucket 2 (7x400 = 2800 total;
        // every other bucket share 0 < 2%). Observed 500 in
        // bucket 0: 500/500 = 100% > 50% in quiet buckets.
        let baseline = flat_baseline(400);
        let observed = stats(8, 500);
        let e = odd_eval(&baseline, observed, [500, 0, 0, 0, 0, 0], [0, 0, 2800, 0, 0, 0]);
    let (findings, _) = evaluate(&e, day(8));
    assert_eq!(signals(&findings), vec![Signal::OddHours]);
    let f = one_finding(&findings, Signal::OddHours);
    assert_eq!(f.observed_value, 1.0);
    // baseline share of the same quiet buckets: bucket 0's
    // baseline count 0 / 500 => 0.0.
    assert_eq!(f.baseline_value, 0.0);
    assert_eq!(f.unit, Unit::Ratio);
}

#[test]
fn odd_hours_at_exact_50_percent_fires() {
    // 250 of 500 in quiet buckets = exactly 50% (>= 50%
    // inclusive): fires. 249 does not.
    let baseline = flat_baseline(400);
    let observed = stats(8, 500);
    let e = odd_eval(&baseline, observed, [250, 0, 250, 0, 0, 0], [0, 0, 2800, 0, 0, 0]);
    let (findings, _) = evaluate(&e, day(8));
    let f = one_finding(&findings, Signal::OddHours);
    assert_eq!(f.observed_value, 0.5);
    // baseline share of the same quiet buckets: bucket 0's
    // baseline count 0 / 500 => 0.0.
    assert_eq!(f.baseline_value, 0.0);
    let e = odd_eval(&baseline, observed, [249, 0, 251, 0, 0, 0], [0, 0, 2800, 0, 0, 0]);
    let (findings, _) = evaluate(&e, day(8));
    assert!(findings.iter().all(|f| f.signal != Signal::OddHours));
}

    #[test]
    fn odd_hours_just_over_50_percent_fires() {
        // 251 of 500 in quiet buckets.
        let baseline = flat_baseline(400);
        let observed = stats(8, 500);
        let e = odd_eval(&baseline, observed, [251, 0, 249, 0, 0, 0], [0, 0, 2800, 0, 0, 0]);
        let (findings, _) = evaluate(&e, day(8));
        assert!(findings.iter().any(|f| f.signal == Signal::OddHours));
    }

    #[test]
    fn odd_hours_below_200_day_requests_does_not_fire() {
        // Share 100% but only 199 requests on the day.
        let baseline = flat_baseline(400);
        let observed = stats(8, 199);
        let e = odd_eval(&baseline, observed, [199, 0, 0, 0, 0, 0], [0, 0, 2800, 0, 0, 0]);
        let (findings, _) = evaluate(&e, day(8));
        assert!(findings.iter().all(|f| f.signal != Signal::OddHours));
    }

    #[test]
    fn odd_hours_at_exactly_200_day_requests_fires() {
        let baseline = flat_baseline(400);
        let observed = stats(8, 200);
        let e = odd_eval(&baseline, observed, [200, 0, 0, 0, 0, 0], [0, 0, 2800, 0, 0, 0]);
        let (findings, _) = evaluate(&e, day(8));
        assert!(findings.iter().any(|f| f.signal == Signal::OddHours));
    }

    #[test]
    fn odd_hours_quiet_bucket_boundary_is_strictly_under_2_percent() {
        // Baseline 7x300 = 2100; a baseline bucket with exactly
        // 2100/50 = 42 requests is exactly 2% (not quiet); 41 is
        // under (quiet).
        let baseline = flat_baseline(300);
        let observed = stats(8, 199);
        // bucket 0 at exactly 2%: not quiet => no odd requests
        let e = odd_eval(&baseline, observed, [199, 0, 0, 0, 0, 0], [42, 0, 2058, 0, 0, 0]);
        let (findings, _) = evaluate(&e, day(8));
        assert!(findings.iter().all(|f| f.signal != Signal::OddHours));
        // bucket 0 at 41/2100 < 2%: quiet => 200/200 odd fires
        let observed = stats(8, 200);
        let e = odd_eval(&baseline, observed, [200, 0, 0, 0, 0, 0], [41, 0, 2059, 0, 0, 0]);
        let (findings, _) = evaluate(&e, day(8));
        assert!(findings.iter().any(|f| f.signal == Signal::OddHours));
    }

    #[test]
    fn odd_hours_share_is_request_weighted_over_day_total() {
        // 400 in a quiet bucket of 1000 day requests = 40% < 50%:
        // no fire — the share is of the day total, not
        // per-bucket.
        let baseline = flat_baseline(400);
        let observed = stats(8, 1_000);
        let e = odd_eval(&baseline, observed, [400, 0, 600, 0, 0, 0], [0, 0, 2800, 0, 0, 0]);
        let (findings, _) = evaluate(&e, day(8));
        assert!(findings.iter().all(|f| f.signal != Signal::OddHours));
    }

    #[test]
    fn odd_hours_histogram_disagreeing_with_day_row_stays_dormant() {
        // hist total 400 != day requests 500: treat as absent.
        let baseline = flat_baseline(400);
        let observed = stats(8, 500);
        let e = odd_eval(&baseline, observed, [400, 0, 0, 0, 0, 0], [0, 0, 2800, 0, 0, 0]);
        let (findings, _) = evaluate(&e, day(8));
        assert!(findings.iter().all(|f| f.signal != Signal::OddHours));
    }

    #[test]
    fn odd_hours_without_histogram_records_no_behavioral_data() {
        // histogram absent + otherwise-eligible day => skip.
        let baseline = flat_baseline(400);
        let observed = stats(8, 500);
        let (findings, skips) = evaluate(&eval(baseline, observed), day(8));
        assert!(findings.iter().all(|f| f.signal != Signal::OddHours));
        assert!(skips.iter().any(|s| s.reason == SkipReason::NoBehavioralData));
    }

    // ---- new_model ----

    #[test]
    fn new_model_fires_at_10_requests_to_absent_model() {
        let baseline = flat_baseline(100);
        let observed = stats(8, 500);
        let e = KeyEvaluation {
            observed_models: vec![KeyModelUsage { model: "gpt-9".to_string(), requests: 10 }],
            baseline_models: Some(vec!["gpt-4".to_string()]),
            ..eval(baseline, observed)
        };
        let (findings, _) = evaluate(&e, day(8));
        assert_eq!(signals(&findings), vec![Signal::NewModel]);
        let f = one_finding(&findings, Signal::NewModel);
        assert_eq!(f.observed_value, 10.0);
        assert_eq!(f.detail, "gpt-9");
    }

    #[test]
    fn new_model_at_9_requests_does_not_fire() {
        let baseline = flat_baseline(100);
        let observed = stats(8, 500);
        let e = KeyEvaluation {
            observed_models: vec![KeyModelUsage { model: "gpt-9".to_string(), requests: 9 }],
            baseline_models: Some(vec!["gpt-4".to_string()]),
            ..eval(baseline, observed)
        };
        let (findings, _) = evaluate(&e, day(8));
        assert!(findings.is_empty());
    }

    #[test]
    fn new_model_present_in_baseline_does_not_fire() {
        let baseline = flat_baseline(100);
        let observed = stats(8, 500);
        let e = KeyEvaluation {
            observed_models: vec![KeyModelUsage { model: "gpt-4".to_string(), requests: 10_000 }],
            baseline_models: Some(vec!["gpt-4".to_string()]),
            ..eval(baseline, observed)
        };
        let (findings, _) = evaluate(&e, day(8));
        assert!(findings.is_empty());
    }

#[test]
fn new_model_threshold_is_per_model_not_summed() {
    // Ten unseen models with 1 request each: no single model
    // reaches >=10, so nothing fires even though the summed
    // count across unseen models is 10.
    let baseline = flat_baseline(100);
    let observed = stats(8, 500);
    let e = KeyEvaluation {
        observed_models: (0..10)
            .map(|n| KeyModelUsage { model: format!("m{n}"), requests: 1 })
            .collect(),
        baseline_models: Some(vec!["gpt-4".to_string()]),
        ..eval(baseline, observed)
    };
    let (findings, _) = evaluate(&e, day(8));
    assert!(findings.iter().all(|f| f.signal != Signal::NewModel));
}

#[test]
fn new_model_multiple_qualifying_models_collapse_to_the_busiest() {
    // Three absent models, two with >=10 requests: exactly ONE
    // finding — the busiest qualifying model ("b", 15 requests
    // beats "a"'s 10); "c" never qualifies (9 < 10).
    let baseline = flat_baseline(100);
    let observed = stats(8, 500);
    let e = KeyEvaluation {
        observed_models: vec![
            KeyModelUsage { model: "a".to_string(), requests: 10 },
            KeyModelUsage { model: "b".to_string(), requests: 15 },
            KeyModelUsage { model: "c".to_string(), requests: 9 },
        ],
        baseline_models: Some(vec!["gpt-4".to_string()]),
        ..eval(baseline, observed)
    };
    let (findings, _) = evaluate(&e, day(8));
    let new_model: Vec<&Finding> =
        findings.iter().filter(|f| f.signal == Signal::NewModel).collect();
    assert_eq!(new_model.len(), 1);
    assert_eq!(new_model[0].detail, "b");
    assert_eq!(new_model[0].observed_value, 15.0);
}

#[test]
fn new_model_tie_breaks_lexicographically() {
    // Two absent models tied at 15 requests: the deterministic
    // winner is the lexicographically smaller model name.
    let baseline = flat_baseline(100);
    let observed = stats(8, 500);
    let e = KeyEvaluation {
        observed_models: vec![
            KeyModelUsage { model: "zzz".to_string(), requests: 15 },
            KeyModelUsage { model: "aaa".to_string(), requests: 15 },
        ],
        baseline_models: Some(vec!["gpt-4".to_string()]),
        ..eval(baseline, observed)
    };
    let (findings, _) = evaluate(&e, day(8));
    let new_model: Vec<&Finding> =
        findings.iter().filter(|f| f.signal == Signal::NewModel).collect();
    assert_eq!(new_model.len(), 1);
    assert_eq!(new_model[0].detail, "aaa");
    assert_eq!(new_model[0].observed_value, 15.0);
}

    #[test]
    fn new_model_without_baseline_models_records_no_behavioral_data() {
        let baseline = flat_baseline(100);
        let observed = stats(8, 500);
        let (findings, skips) = evaluate(&eval(baseline, observed), day(8));
        assert!(findings.is_empty());
        assert!(skips.iter().any(|s| s.reason == SkipReason::NoBehavioralData));
    }

    // ---- probes ----

    #[test]
    fn probes_fires_at_50_failures_and_5x_baseline() {
        // baseline 10/day failures; observed 51 > 5x10 with >=50.
        let baseline: Vec<KeyDailyStats> = (1..=BASELINE_WINDOW_LEN as u32)
            .map(|n| KeyDailyStats { validation_failures: 10, ..stats(n, 1_000) })
            .collect();
        let observed = KeyDailyStats { validation_failures: 51, ..stats(8, 1_000) };
        let (findings, skips) = evaluate(&eval(baseline, observed), day(8));
        assert_eq!(signals(&findings), vec![Signal::Probes]);
        assert_no_data_skips_only(&skips);
        let f = one_finding(&findings, Signal::Probes);
        assert_eq!(f.observed_value, 51.0);
        assert_eq!(f.baseline_value, 10.0);
    }

    #[test]
    fn probes_at_49_failures_does_not_fire() {
        let baseline: Vec<KeyDailyStats> = (1..=BASELINE_WINDOW_LEN as u32)
            .map(|n| KeyDailyStats { validation_failures: 10, ..stats(n, 1_000) })
            .collect();
        let observed = KeyDailyStats { validation_failures: 49, ..stats(8, 1_000) };
        let (findings, skips) = evaluate(&eval(baseline, observed), day(8));
        assert!(findings.is_empty());
        assert_no_data_skips_only(&skips);
    }

    #[test]
    fn probes_chronically_failing_key_stays_quiet() {
        // Baseline 100 failures/day; observed 350 = 3.5x < 5x
        // even though >=50 absolute: the relative gate holds.
        let baseline: Vec<KeyDailyStats> = (1..=BASELINE_WINDOW_LEN as u32)
            .map(|n| KeyDailyStats { validation_failures: 100, ..stats(n, 1_000) })
            .collect();
        let observed = KeyDailyStats { validation_failures: 350, ..stats(8, 1_000) };
        let (findings, _) = evaluate(&eval(baseline, observed), day(8));
        assert!(findings.is_empty());
    }

#[test]
fn probes_relative_gate_is_inclusive_at_exact_5x() {
    // 500 = exactly 5x baseline 100 (>=5x inclusive) with
    // >=50 absolute: fires; 499 does not. Both adjacent
    // points checked.
    let baseline: Vec<KeyDailyStats> = (1..=BASELINE_WINDOW_LEN as u32)
        .map(|n| KeyDailyStats { validation_failures: 100, ..stats(n, 1_000) })
        .collect();
    let observed = KeyDailyStats { validation_failures: 500, ..stats(8, 1_000) };
    let (findings, _) = evaluate(&eval(baseline.clone(), observed), day(8));
    assert_eq!(signals(&findings), vec![Signal::Probes]);
    let observed = KeyDailyStats { validation_failures: 499, ..stats(8, 1_000) };
    let (findings, _) = evaluate(&eval(baseline, observed), day(8));
    assert!(findings.is_empty());
}

    #[test]
    fn probes_absolute_leg_fires_on_zero_baseline() {
        // Zero baseline traffic: ratio undefined, absolute leg
        // (>=50) still fires per design; the day also qualifies
        // for new_activity (25+ requests, zero baseline).
        let baseline = flat_baseline(0);
        let observed = KeyDailyStats { validation_failures: 50, ..stats(8, 600) };
        let (findings, skips) = evaluate(&eval(baseline, observed), day(8));
        assert!(findings.iter().any(|f| f.signal == Signal::Probes));
        assert!(reasons(&skips).contains(&SkipReason::ZeroBaseline));
        let f = one_finding(&findings, Signal::Probes);
        assert_eq!(f.baseline_value, 0.0);
    }

    #[test]
    fn probes_absolute_leg_at_49_on_zero_baseline_does_not_fire() {
        // 49 failures: under the absolute leg, so no probes
        // finding (new_activity still fires independently).
        let baseline = flat_baseline(0);
        let observed = KeyDailyStats { validation_failures: 49, ..stats(8, 600) };
        let (findings, _) = evaluate(&eval(baseline, observed), day(8));
        assert!(!findings.iter().any(|f| f.signal == Signal::Probes));
    }

    // ---- skip reasons and day handling ----

    #[test]
    fn fewer_than_four_baseline_traffic_days_skips_ratio_signals() {
        // 3 of 7 days with traffic.
        let mut baseline = flat_baseline(0);
        for b in baseline.iter_mut().take(3) {
            b.requests = 1_000;
        }
        let observed = stats(8, 50_000);
        let (findings, skips) = evaluate(&eval(baseline, observed), day(8));
        assert!(findings.is_empty());
        assert_eq!(reasons(&skips), vec![SkipReason::InsufficientBaselineDays]);
    }

    #[test]
    fn exactly_four_baseline_traffic_days_passes_the_gate() {
        let mut baseline = flat_baseline(0);
        for b in baseline.iter_mut().take(4) {
            b.requests = 1_000;
        }
        // 1000x4/7 ~ 571/day; 20000 observed ~ 35x with 19429 over
        let observed = stats(8, 20_000);
        let (findings, skips) = evaluate(&eval(baseline, observed), day(8));
        assert!(findings.iter().any(|f| f.signal == Signal::Traffic));
        assert!(!reasons(&skips).contains(&SkipReason::InsufficientBaselineDays));
    }

    #[test]
    fn zero_requests_on_evaluated_day_skips_all() {
        let baseline = flat_baseline(1_000);
        let observed = stats(8, 0);
        let (findings, skips) = evaluate(&eval(baseline, observed), day(8));
        assert!(findings.is_empty());
        assert_eq!(reasons(&skips), vec![SkipReason::NoTrafficOnEvaluatedDay]);
    }

    #[test]
    fn mismatched_observed_day_counts_as_no_traffic() {
        // observed row bucketed on a different day than the
        // evaluated day: day_requests reads 0.
        let baseline = flat_baseline(1_000);
        let observed = stats(9, 50_000);
        let (findings, skips) = evaluate(&eval(baseline, observed), day(8));
        assert!(findings.is_empty());
        assert_eq!(reasons(&skips), vec![SkipReason::NoTrafficOnEvaluatedDay]);
    }

#[test]
fn baseline_day_outside_window_rejected_as_malformed() {
    // 7 rows, none equal to the observed day, but one bucketed
    // 30 days before the evaluated day: window mismatch.
    let mut baseline = flat_baseline(1_000);
    baseline[0].day = day(8) - chrono::Days::new(30);
    let observed = stats(8, 50_000);
    let (findings, skips) = evaluate(&eval(baseline, observed), day(8));
    assert!(findings.is_empty());
    assert_eq!(reasons(&skips), vec![SkipReason::MalformedBaseline]);
}

#[test]
fn baseline_including_evaluated_day_rejected_as_malformed() {
    // A baseline row bucketed on the evaluated day itself
    // double counts: rejected.
    let mut baseline = flat_baseline(1_000);
    baseline[6].day = day(8);
    let observed = stats(8, 50_000);
    let (findings, skips) = evaluate(&eval(baseline, observed), day(8));
    assert!(findings.is_empty());
    assert_eq!(reasons(&skips), vec![SkipReason::MalformedBaseline]);
}

#[test]
fn baseline_with_duplicate_days_rejected_as_malformed() {
    // 7 rows, all inside the window, none on the evaluated
    // day — but one day appears twice (another day missing):
    // duplicate window coverage is malformed.
    let mut baseline = flat_baseline(1_000);
    baseline[6].day = baseline[0].day;
    let observed = stats(8, 50_000);
    let (findings, skips) = evaluate(&eval(baseline, observed), day(8));
    assert!(findings.is_empty());
    assert_eq!(reasons(&skips), vec![SkipReason::MalformedBaseline]);
}

#[test]
fn wrong_baseline_len_rejected_as_malformed() {
    let baseline: Vec<KeyDailyStats> = (1..=6).map(|n| stats(n, 1_000)).collect();
    let observed = stats(8, 50_000);
    let (findings, skips) = evaluate(&eval(baseline, observed), day(8));
    assert!(findings.is_empty());
    assert_eq!(reasons(&skips), vec![SkipReason::MalformedBaseline]);
}

    #[test]
    fn no_traffic_day_with_insufficient_baseline_records_both_skips() {
        let mut baseline = flat_baseline(0);
        for b in baseline.iter_mut().take(2) {
            b.requests = 100;
        }
        let observed = stats(8, 0);
        let (findings, skips) = evaluate(&eval(baseline, observed), day(8));
        assert!(findings.is_empty());
        assert_eq!(
            reasons(&skips),
            vec![
                SkipReason::NoTrafficOnEvaluatedDay,
                SkipReason::InsufficientBaselineDays,
            ]
        );
    }
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    fn any_date() -> impl Strategy<Value = NaiveDate> {
        (0u32..4_000u32).prop_map(|n| {
            chrono::NaiveDate::from_ymd_opt(2026, 1, 1)
                .unwrap()
                .checked_add_days(chrono::Days::new(n as u64))
                .unwrap()
        })
    }

    fn key() -> KeyRef {
        KeyRef {
            key_id: "k1".to_string(),
            name: Some("prod".to_string()),
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 64,
            .. ProptestConfig::default()
        })]

        /// Purity: identical inputs yield identical findings
        /// and skips (Req 2.4).
        #[test]
        fn prop_evaluate_is_pure(
            d in any_date(),
            seed in 0i64..2_000_000_000,
            o_req in 0i64..1_000_000,
            o_fail in 0i64..100_000,
        ) {
            let mk = || KeyEvaluation {
                key: key(),
                observed: KeyDailyStats {
                    day: d,
                    cost_micro_usd: seed,
                    requests: o_req,
                    validation_failures: o_fail,
                    input_tokens: seed,
                    output_tokens: o_req,
                },
                baseline: (1..=BASELINE_WINDOW_LEN as u32)
                    .map(|n| KeyDailyStats {
                        day: d - chrono::Days::new(n as u64),
                        cost_micro_usd: seed % 100_000_000,
                        requests: (seed / 7) % 10_000,
                        validation_failures: seed % 700,
                        input_tokens: seed % 500_000,
                        output_tokens: seed % 500_000,
                    })
                    .collect(),
                hour_histogram: None,
                baseline_hour_histogram: None,
                observed_models: Vec::new(),
                baseline_models: None,
            };
            let (f1, s1) = evaluate(&mk(), d);
            let (f2, s2) = evaluate(&mk(), d);
            prop_assert_eq!(f1, f2);
            prop_assert_eq!(s1, s2);
        }

        /// Monotonicity: higher observed spend / traffic /
        /// tokens never yield fewer findings of that signal.
        #[test]
        fn prop_higher_observed_never_fewer_findings(
            d in any_date(),
            base_cost in 0i64..200_000_000,
            extra_cost in 1i64..200_000_000,
            base_req in 1_000i64..10_000,
            base_tok in 0i64..2_000_000,
            extra_tok in 1i64..2_000_000,
        ) {
            let baseline: Vec<KeyDailyStats> = (1..=BASELINE_WINDOW_LEN as u32)
                .map(|n| KeyDailyStats {
                    day: d - chrono::Days::new(n as u64),
                    cost_micro_usd: 10_000_000,
                    requests: 1_000,
                    validation_failures: 0,
                    input_tokens: 100_000,
                    output_tokens: 0,
                })
                .collect();
            let mk = |cost: i64, req: i64, tok: i64| KeyEvaluation {
                key: key(),
                observed: KeyDailyStats {
                    day: d,
                    cost_micro_usd: cost,
                    requests: req,
                    validation_failures: 0,
                    input_tokens: tok,
                    output_tokens: 0,
                },
                baseline: baseline.clone(),
                hour_histogram: None,
                baseline_hour_histogram: None,
                observed_models: Vec::new(),
                baseline_models: None,
            };
            let lo = evaluate(&mk(base_cost, base_req, base_tok), d).0;
            let hi = evaluate(&mk(base_cost + extra_cost, base_req, base_tok), d).0;
            prop_assert!(hi.iter().filter(|f| f.signal == Signal::Spend).count()
                >= lo.iter().filter(|f| f.signal == Signal::Spend).count());
            let lo = evaluate(&mk(base_cost, base_req, base_tok), d).0;
            let hi = evaluate(&mk(base_cost, base_req * 10 + 1_000, base_tok), d).0;
            prop_assert!(hi.iter().filter(|f| f.signal == Signal::Traffic).count()
                >= lo.iter().filter(|f| f.signal == Signal::Traffic).count());
            let lo = evaluate(&mk(base_cost, base_req, base_tok), d).0;
            let hi = evaluate(&mk(base_cost, base_req, base_tok + extra_tok), d).0;
            prop_assert!(hi.iter().filter(|f| f.signal == Signal::Tokens).count()
                >= lo.iter().filter(|f| f.signal == Signal::Tokens).count());
        }

/// No-panic: arbitrary (degenerate: zero, huge,
/// mismatched dates, wrong lengths) aggregates never
/// panic, finding values stay finite, a mismatched
/// observed day reads as no-traffic and a wrong baseline
/// length as a malformed baseline. Validation failures
/// never exceed their day's requests (real-ledger
/// constraint).
        #[test]
        fn prop_no_panic_on_arbitrary_aggregates(
            d in any_date(),
            c0 in 0i64..2_000_000_000, c1 in 0i64..2_000_000_000, c2 in 0i64..2_000_000_000, c3 in 0i64..2_000_000_000,
            c4 in 0i64..2_000_000_000, c5 in 0i64..2_000_000_000, c6 in 0i64..2_000_000_000,
            r0 in 0i64..1_000_000, r1 in 0i64..1_000_000, r2 in 0i64..1_000_000, r3 in 0i64..1_000_000,
            r4 in 0i64..1_000_000, r5 in 0i64..1_000_000, r6 in 0i64..1_000_000,
            v0 in 0i64..100_000, v1 in 0i64..100_000, v2 in 0i64..100_000, v3 in 0i64..100_000,
            v4 in 0i64..100_000, v5 in 0i64..100_000, v6 in 0i64..100_000,
            t0 in 0i64..2_000_000_000, t1 in 0i64..2_000_000_000, t2 in 0i64..2_000_000_000,
            t3 in 0i64..2_000_000_000, t4 in 0i64..2_000_000_000, t5 in 0i64..2_000_000_000,
            t6 in 0i64..2_000_000_000,
            o_cost in 0i64..2_000_000_000,
            o_req in 0i64..1_000_000,
            o_fail_raw in 0i64..100_000,
            o_tok in 0i64..2_000_000_000,
            mismatched_day in proptest::bool::ANY,
            wrong_len in proptest::bool::ANY,
        ) {
            let costs = [c0, c1, c2, c3, c4, c5, c6];
            let reqs = [r0, r1, r2, r3, r4, r5, r6];
            let fails = [v0, v1, v2, v3, v4, v5, v6];
            let toks = [t0, t1, t2, t3, t4, t5, t6];
            // failures are a subset of requests per real ledgers
            let o_fail = o_fail_raw.min(o_req);
            let baseline: Vec<KeyDailyStats> = (0..if wrong_len { 5 } else { 7 })
                .map(|n| KeyDailyStats {
                    day: d - chrono::Days::new(n as u64 + 1),
                    cost_micro_usd: costs[n],
                    requests: reqs[n],
                    validation_failures: fails[n].min(reqs[n]),
                    input_tokens: toks[n],
                    output_tokens: 0,
                })
                .collect();
            let observed = KeyDailyStats {
                day: if mismatched_day { d + chrono::Days::new(3) } else { d },
                cost_micro_usd: o_cost,
                requests: o_req,
                validation_failures: o_fail,
                input_tokens: o_tok,
                output_tokens: 0,
            };
            let e = KeyEvaluation {
                key: key(),
                observed,
                baseline,
                hour_histogram: None,
                baseline_hour_histogram: None,
                observed_models: Vec::new(),
                baseline_models: None,
            };
    let (findings, skips) = evaluate(&e, d);
    if mismatched_day || wrong_len {
        prop_assert!(findings.is_empty());
        let expected = if wrong_len {
            SkipReason::MalformedBaseline
        } else {
            SkipReason::NoTrafficOnEvaluatedDay
        };
        prop_assert!(skips.iter().any(|s| s.reason == expected));
    }
            for f in &findings {
                prop_assert!(f.observed_value.is_finite());
                prop_assert!(f.baseline_value.is_finite());
            }
        }

        /// Hostile-name no-secret invariant (NFR security): key
        /// display names carrying secret-shaped substrings
        /// never leak into finding details or thresholds —
        /// findings carry the key id in `key.key_id`; the name
        /// only ever appears in the cosmetic `key.name` field.
        #[test]
        fn prop_hostile_key_names_never_leak_beyond_name_field(
            d in any_date(),
            hostile in proptest::sample::select(vec![
                "sk-proj-abc123".to_string(),
                "vk_live_secret9".to_string(),
                "sk-".to_string(),
                "bearer sk-000".to_string(),
            ]),
            o_req in 0i64..50_000,
            o_cost in 0i64..200_000_000,
        ) {
            let e = KeyEvaluation {
                key: KeyRef { key_id: "kid-1".to_string(), name: Some(hostile) },
                observed: KeyDailyStats {
                    day: d,
                    cost_micro_usd: o_cost,
                    requests: o_req,
                    validation_failures: 0,
                    input_tokens: 0,
                    output_tokens: 0,
                },
                baseline: (1..=BASELINE_WINDOW_LEN as u32)
                    .map(|n| KeyDailyStats {
                        day: d - chrono::Days::new(n as u64),
                        cost_micro_usd: 10_000_000,
                        requests: 1_000,
                        validation_failures: 0,
                        input_tokens: 100_000,
                        output_tokens: 0,
                    })
                    .collect(),
                hour_histogram: None,
                baseline_hour_histogram: None,
                observed_models: Vec::new(),
                baseline_models: None,
            };
            let (findings, _skips) = evaluate(&e, d);
            for f in &findings {
                prop_assert_eq!(&f.key.key_id, "kid-1");
                // detail / threshold never contain the hostile name
                prop_assert!(!f.detail.contains("sk-") && !f.detail.contains("vk_"));
                prop_assert!(!f.threshold.contains("sk-") && !f.threshold.contains("vk_"));
                // the only place the name may appear is f.key.name
                if let Some(n) = &f.key.name {
                    prop_assert!(n.contains("sk-") || n.contains("vk_"));
                }
            }
        }
    }
}
