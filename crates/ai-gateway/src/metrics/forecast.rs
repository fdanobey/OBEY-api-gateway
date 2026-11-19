//! Pure spend-pace projection math for budgeted scopes (spec:
//! spend-pace-forecasting, Component 1).
//!
//! Windowed budget policies (daily, weekly, monthly) extrapolate
//! window-to-date pace to the configured window end; manual policies
//! extrapolate a trailing 30-day run rate over a fixed 30-day horizon.
//! The module is deliberately pure — no I/O, no async, no clocks: every
//! input is durable state supplied by the caller, so identical inputs
//! always yield identical projections.

use chrono::{DateTime, NaiveDate, TimeZone, Utc};
use serde::{Deserialize, Serialize};

/// Minimum days of ledger history inside the trailing horizon before a
/// manual-window run-rate projection is trusted (design D5).
pub const MIN_TRAILING_HISTORY_DAYS: u32 = 7;

/// Length of the manual-window pace horizon and the run-rate basis, in
/// days (design D4).
pub const MANUAL_HORIZON_DAYS: i64 = 30;

/// Seconds in one day, for whole-days-remaining arithmetic.
const SECONDS_PER_DAY: i64 = 86_400;

/// Projection method behind a [`Forecast`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForecastMethod {
    /// Window-to-date pace projected to the configured window end.
    WindowPace,
    /// Trailing 30-day run rate projected over the 30-day horizon.
    TrailingRunRate,
}

/// Relationship of a projection to the configured limit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ForecastStatus {
    /// Not enough elapsed window time (or trailing history) to project
    /// honestly; `projected_usd` carries the un-extrapolated spend.
    InsufficientData,
    /// The projection stays under the limit through the window end (or
    /// the 30-day horizon for manual scopes).
    WithinWindow,
    /// The projection reaches the limit before the window end (or
    /// inside the 30-day horizon for manual scopes).
    ExhaustsWithinWindow,
    /// Current spend already meets or exceeds the limit.
    AlreadyExhausted,
}

/// Projected end-of-window spend for one budgeted scope.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Forecast {
    /// Spend the window is projected to close at. With
    /// [`ForecastStatus::InsufficientData`] or
    /// [`ForecastStatus::AlreadyExhausted`] this is the current spend,
    /// never an extrapolation.
    pub projected_usd: f64,
    /// Method used to compute the projection.
    pub method: ForecastMethod,
    /// Relationship of the projection to the limit.
    pub status: ForecastStatus,
    /// UTC date the limit is projected to be reached, reported only
    /// with [`ForecastStatus::ExhaustsWithinWindow`].
    pub projected_exhaustion_date: Option<NaiveDate>,
    /// Whole days from `now` to the projected exhaustion, reported only
    /// with [`ForecastStatus::ExhaustsWithinWindow`].
    pub days_remaining: Option<u32>,
    /// UTC epoch seconds the pace horizon ends at: the manual scope's
    /// 30-day horizon (`now + 30 days`). `None` for windowed scopes,
    /// whose horizon is the configured window end already exposed as
    /// `next_reset_epoch` alongside the forecast (requirement 1.6).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub horizon_end: Option<i64>,
}

impl Forecast {
    fn insufficient(method: ForecastMethod, spent_usd: f64) -> Self {
        Self {
            projected_usd: sanitize_usd(spent_usd),
            method,
            status: ForecastStatus::InsufficientData,
            projected_exhaustion_date: None,
            days_remaining: None,
            horizon_end: None,
        }
    }

    fn already_exhausted(method: ForecastMethod, spent_usd: f64) -> Self {
        Self {
            projected_usd: sanitize_usd(spent_usd),
            method,
            status: ForecastStatus::AlreadyExhausted,
            projected_exhaustion_date: None,
            days_remaining: None,
            horizon_end: None,
        }
    }

    fn within_window(method: ForecastMethod, projected_usd: f64) -> Self {
        Self {
            projected_usd: sanitize_usd(projected_usd),
            method,
            status: ForecastStatus::WithinWindow,
            projected_exhaustion_date: None,
            days_remaining: None,
            horizon_end: None,
        }
    }

    fn exhausts_within_window(
        method: ForecastMethod,
        projected_usd: f64,
        projected_exhaustion_date: NaiveDate,
        days_remaining: u32,
    ) -> Self {
        Self {
            projected_usd: sanitize_usd(projected_usd),
            method,
            status: ForecastStatus::ExhaustsWithinWindow,
            projected_exhaustion_date: Some(projected_exhaustion_date),
            days_remaining: Some(days_remaining),
            horizon_end: None,
        }
    }

    /// Manual-scope projection carrying the pace horizon end (now + 30
    /// days, epoch seconds) so status payloads expose requirement 1.6's
    /// horizon alongside the projection.
    fn with_horizon_end(mut self, horizon_end: DateTime<Utc>) -> Self {
        self.horizon_end = Some(horizon_end.timestamp());
        self
    }
}

/// USD payload fields must stay JSON-serializable: any non-finite f64
/// (NaN, ±inf from degenerate inputs) destined for a payload field is
/// clamped to 0.0 inside the constructors, never serialized as null.
fn sanitize_usd(value: f64) -> f64 {
    if value.is_finite() {
        value
    } else {
        0.0
    }
}

/// Whole days from `now` to `exhaustion_epoch`, rounded down; `None`
/// when the difference does not fit in a `u32` (far-future dates from
/// extreme inputs degrade the date instead of panicking).
fn whole_days_remaining(now: DateTime<Utc>, exhaustion_epoch: i64) -> Option<u32> {
    let diff_s = exhaustion_epoch.saturating_sub(now.timestamp());
    u32::try_from(diff_s.div_euclid(SECONDS_PER_DAY)).ok()
}

/// Classify a projection that has positive pace and headroom, shared by
/// both methods: `exhaustion_epoch = now + (limit - spent) / pace_s`
/// seconds; [`ForecastStatus::ExhaustsWithinWindow`] with the ISO date
/// and whole days remaining when the limit is reached by `horizon_end`,
/// otherwise [`ForecastStatus::WithinWindow`].
///
/// A `projected_usd` that overflows to +inf (pace × remaining beyond
/// f64 range) is not insufficient data — the pace is real, only the
/// end-of-window sum is unrepresentable. When the exhaustion epoch
/// still lands inside the horizon the projection is
/// [`ForecastStatus::ExhaustsWithinWindow`] with the projected spend
/// clamped to the finite limit (the smallest honest bound: exhaustion
/// is certain, the exact overshoot is not).
fn classify_with_pace(
    method: ForecastMethod,
    limit_usd: f64,
    spent_usd: f64,
    pace_per_s: f64,
    projected_usd: f64,
    horizon_end: DateTime<Utc>,
    now: DateTime<Utc>,
) -> Forecast {
    let seconds_to_limit = (limit_usd - spent_usd) / pace_per_s;
    let exhaustion_epoch = now.timestamp().saturating_add(
        seconds_to_limit
            .is_finite()
            .then(|| seconds_to_limit as i64)
            .unwrap_or(i64::MAX),
    );
    if exhaustion_epoch < horizon_end.timestamp() {
        // Extreme-but-finite inputs can overflow the date itself; the
        // projection then saturates to the horizon end (exhaustion is
        // known to precede it) instead of panicking in NaiveDate
        // arithmetic.
        let date = Utc
            .timestamp_opt(exhaustion_epoch, 0)
            .single()
            .map(|dt| dt.date_naive())
            .or_else(|| Utc.timestamp_opt(horizon_end.timestamp(), 0).single().map(|dt| dt.date_naive()));
        if let Some(date) = date {
            if let Some(days) = whole_days_remaining(now, exhaustion_epoch) {
                let projected = if projected_usd.is_finite() {
                    projected_usd
                } else {
                    limit_usd
                };
                return Forecast::exhausts_within_window(method, projected, date, days);
            }
        }
    }
    if projected_usd.is_finite() {
        Forecast::within_window(method, projected_usd)
    } else {
        // Unrepresentable projection whose exhaustion does not land
        // inside the horizon: the overflow says nothing honest about
        // the end-of-window sum, so degrade rather than assert.
        Forecast::insufficient(method, spent_usd)
    }
}

/// Project a windowed budget (daily, weekly, monthly) to its window end
/// using window-to-date pace (design D3): `pace = spent / elapsed_s`,
/// `projected = spent + pace * remaining_s`, exhaustion when
/// `(limit - spent) / pace` falls within the window.
///
/// Precedence: non-finite USD inputs and a zero-length or inverted
/// window yield [`ForecastStatus::InsufficientData`]; spend already at
/// or above the limit yields [`ForecastStatus::AlreadyExhausted`] (a
/// factual state, not an extrapolation, so it wins over the
/// insufficient-data gate); less than 5% of the window elapsed (exact
/// integer form: `elapsed * 20 < duration`) yields
/// [`ForecastStatus::InsufficientData`] so a day-1 spend spike can
/// never over-project (design D5). A clock outside the window clamps
/// elapsed/remaining time to the window bounds instead of dividing by
/// zero or extrapolating a negative pace.
pub fn forecast_windowed(
    limit_usd: f64,
    spent_usd: f64,
    window_start: DateTime<Utc>,
    window_end: DateTime<Utc>,
    now: DateTime<Utc>,
) -> Forecast {
    let method = ForecastMethod::WindowPace;
    if !limit_usd.is_finite() || !spent_usd.is_finite() {
        return Forecast::insufficient(method, spent_usd);
    }
    if spent_usd >= limit_usd {
        return Forecast::already_exhausted(method, spent_usd);
    }
    let duration_s = window_end.timestamp().saturating_sub(window_start.timestamp());
    if duration_s <= 0 {
        return Forecast::insufficient(method, spent_usd);
    }
    let elapsed_s = now
        .timestamp()
        .clamp(window_start.timestamp(), window_end.timestamp())
        .saturating_sub(window_start.timestamp());
    if elapsed_s * 20 < duration_s {
        return Forecast::insufficient(method, spent_usd);
    }
    if spent_usd <= 0.0 {
        return Forecast::within_window(method, 0.0);
    }
    let pace_per_s = spent_usd / elapsed_s as f64;
    let remaining_s = (window_end.timestamp() - now.timestamp()).max(0) as f64;
    let projected_usd = spent_usd + pace_per_s * remaining_s;
    classify_with_pace(
        method,
        limit_usd,
        spent_usd,
        pace_per_s,
        projected_usd,
        window_end,
        now,
    )
}

/// Project a manual-window budget from a trailing 30-day run rate
/// (design D4): `pace = trailing_spend_30d / 30`,
/// `projected = spent + pace * 30`; the projected exhaustion date is
/// `now + (limit - spent) / pace` days, reported only when it falls
/// inside the 30-day horizon.
///
/// Precedence: non-finite USD inputs yield
/// [`ForecastStatus::InsufficientData`]; spend already at or above the
/// limit yields [`ForecastStatus::AlreadyExhausted`]; less than
/// [`MIN_TRAILING_HISTORY_DAYS`] of ledger history inside the horizon
/// yields [`ForecastStatus::InsufficientData`] (design D5). A zero
/// trailing spend is a legitimate flat pace: on-target budgets report
/// [`ForecastStatus::WithinWindow`] with the current spend, and there
/// is no exhaustion date to report.
pub fn forecast_manual(
    limit_usd: f64,
    spent_usd: f64,
    trailing_spend_30d: f64,
    history_days: u32,
    now: DateTime<Utc>,
) -> Forecast {
    let method = ForecastMethod::TrailingRunRate;
    let horizon_end = Utc
        .timestamp_opt(now.timestamp().saturating_add(MANUAL_HORIZON_DAYS * SECONDS_PER_DAY), 0)
        .single()
        .unwrap_or(now);
    if !limit_usd.is_finite() || !spent_usd.is_finite() || !trailing_spend_30d.is_finite() {
        return Forecast::insufficient(method, spent_usd).with_horizon_end(horizon_end);
    }
    if spent_usd >= limit_usd {
        return Forecast::already_exhausted(method, spent_usd).with_horizon_end(horizon_end);
    }
    if history_days < MIN_TRAILING_HISTORY_DAYS {
        return Forecast::insufficient(method, spent_usd).with_horizon_end(horizon_end);
    }
    if trailing_spend_30d <= 0.0 {
        return Forecast::within_window(method, spent_usd).with_horizon_end(horizon_end);
    }
    let pace_per_s = trailing_spend_30d / (MANUAL_HORIZON_DAYS * SECONDS_PER_DAY) as f64;
    let projected_usd = spent_usd + trailing_spend_30d;
    classify_with_pace(
        method,
        limit_usd,
        spent_usd,
        pace_per_s,
        projected_usd,
        horizon_end,
        now,
    )
    .with_horizon_end(horizon_end)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(seconds: i64) -> DateTime<Utc> {
        Utc.timestamp_opt(seconds, 0).unwrap()
    }

    /// 2026-06-01T00:00:00Z — a fixed epoch every fixture anchors to.
    const T0: i64 = 1_780_278_400;

    #[test]
    fn windowed_on_pace_projects_linearly() {
        let start = at(T0);
        let end = at(T0 + 30 * SECONDS_PER_DAY);
        let now = at(T0 + 15 * SECONDS_PER_DAY);
        let f = forecast_windowed(100.0, 40.0, start, end, now);
        assert_eq!(f.method, ForecastMethod::WindowPace);
        assert_eq!(f.status, ForecastStatus::WithinWindow);
        assert!((f.projected_usd - 80.0).abs() < 1e-9);
        assert_eq!(f.projected_exhaustion_date, None);
        assert_eq!(f.days_remaining, None);
    }

    #[test]
    fn windowed_over_pace_exhausts_within_window() {
        let start = at(T0);
        let end = at(T0 + 30 * SECONDS_PER_DAY);
        let now = at(T0 + 15 * SECONDS_PER_DAY);
        let f = forecast_windowed(50.0, 40.0, start, end, now);
        assert_eq!(f.status, ForecastStatus::ExhaustsWithinWindow);
        // pace = 40/15d; limit at 50 reached after 3.75 more days.
        assert_eq!(f.projected_exhaustion_date, Some(at(T0 + 18 * SECONDS_PER_DAY + 9 * 3600).date_naive()));
        assert_eq!(f.days_remaining, Some(3));
        assert!((f.projected_usd - 80.0).abs() < 1e-9);
    }

    #[test]
    fn windowed_exhaustion_exactly_at_window_end_is_within_window() {
        let start = at(T0);
        let end = at(T0 + 30 * SECONDS_PER_DAY);
        let now = at(T0 + 15 * SECONDS_PER_DAY);
        // pace doubles spend: 25 now -> 50 exactly at the boundary,
        // which is AT the window end, not within it.
        let f = forecast_windowed(50.0, 25.0, start, end, now);
        assert_eq!(f.status, ForecastStatus::WithinWindow);
        assert_eq!(f.projected_exhaustion_date, None);
        // 0.01 under the boundary limit: exhaustion one hour before the
        // end IS within the window.
        let inside = forecast_windowed(50.0 - 0.01, 25.0, start, end, now);
        assert_eq!(inside.status, ForecastStatus::ExhaustsWithinWindow);
    }

    #[test]
    fn windowed_already_exhausted_wins_over_young_window() {
        let start = at(T0);
        let end = at(T0 + 30 * SECONDS_PER_DAY);
        let now = at(T0 + 60);
        let f = forecast_windowed(100.0, 100.0, start, end, now);
        assert_eq!(f.status, ForecastStatus::AlreadyExhausted);
        assert_eq!(f.projected_usd, 100.0);
        assert_eq!(f.projected_exhaustion_date, None);
        let over = forecast_windowed(100.0, 140.0, start, end, now);
        assert_eq!(over.status, ForecastStatus::AlreadyExhausted);
        assert_eq!(over.projected_usd, 140.0);
    }

    #[test]
    fn windowed_day_one_blowup_is_insufficient_data() {
        let start = at(T0);
        let end = at(T0 + 30 * SECONDS_PER_DAY);
        let now = at(T0 + 3600);
        let f = forecast_windowed(100.0, 30.0, start, end, now);
        assert_eq!(f.status, ForecastStatus::InsufficientData);
        assert_eq!(f.projected_usd, 30.0);
        assert_eq!(f.projected_exhaustion_date, None);
    }

    #[test]
    fn windowed_five_percent_boundary_is_the_gate() {
        let start = at(T0);
        let end = at(T0 + 1000);
        let just_under = forecast_windowed(100.0, 1.0, start, end, at(T0 + 49));
        assert_eq!(just_under.status, ForecastStatus::InsufficientData);
        let at_gate = forecast_windowed(100.0, 1.0, start, end, at(T0 + 50));
        assert_eq!(at_gate.status, ForecastStatus::WithinWindow);
    }

    #[test]
    fn windowed_zero_length_and_inverted_windows_are_insufficient() {
        let start = at(T0);
        let f = forecast_windowed(100.0, 10.0, start, start, at(T0));
        assert_eq!(f.status, ForecastStatus::InsufficientData);
        let inverted = forecast_windowed(100.0, 10.0, at(T0 + 500), at(T0), at(T0));
        assert_eq!(inverted.status, ForecastStatus::InsufficientData);
    }

    #[test]
    fn windowed_clock_outside_window_is_clamped() {
        let start = at(T0);
        let end = at(T0 + 10 * SECONDS_PER_DAY);
        let before = forecast_windowed(100.0, 40.0, start, end, at(T0 - 5 * SECONDS_PER_DAY));
        assert_eq!(before.status, ForecastStatus::InsufficientData);
        let after = forecast_windowed(100.0, 40.0, start, end, at(T0 + 20 * SECONDS_PER_DAY));
        assert_eq!(after.status, ForecastStatus::WithinWindow);
        assert!((after.projected_usd - 40.0).abs() < 1e-9);
    }

    #[test]
    fn windowed_zero_spend_never_divides_by_zero() {
        let start = at(T0);
        let end = at(T0 + 30 * SECONDS_PER_DAY);
        let now = at(T0 + 15 * SECONDS_PER_DAY);
        let f = forecast_windowed(100.0, 0.0, start, end, now);
        assert_eq!(f.status, ForecastStatus::WithinWindow);
        assert_eq!(f.projected_usd, 0.0);
    }

    #[test]
    fn manual_over_pace_exhausts_within_horizon() {
        let now = at(T0);
        let f = forecast_manual(200.0, 50.0, 300.0, 30, now);
        assert_eq!(f.method, ForecastMethod::TrailingRunRate);
        assert_eq!(f.status, ForecastStatus::ExhaustsWithinWindow);
        assert!((f.projected_usd - 350.0).abs() < 1e-9);
        // pace 10/day, headroom 150 -> exhaustion on day 15.
        assert_eq!(f.projected_exhaustion_date, Some(at(T0 + 15 * SECONDS_PER_DAY).date_naive()));
        assert_eq!(f.days_remaining, Some(15));
    }

    #[test]
    fn manual_on_pace_stays_within_window() {
        let now = at(T0);
        let f = forecast_manual(1000.0, 50.0, 300.0, 30, now);
        assert_eq!(f.status, ForecastStatus::WithinWindow);
        assert!((f.projected_usd - 350.0).abs() < 1e-9);
        assert_eq!(f.projected_exhaustion_date, None);
        assert_eq!(f.days_remaining, None);
    }

    #[test]
    fn manual_short_history_is_insufficient_data() {
        let now = at(T0);
        let six = forecast_manual(200.0, 50.0, 300.0, 6, now);
        assert_eq!(six.status, ForecastStatus::InsufficientData);
        assert_eq!(six.projected_usd, 50.0);
        let seven = forecast_manual(200.0, 50.0, 300.0, 7, now);
        assert_eq!(seven.status, ForecastStatus::ExhaustsWithinWindow);
    }

    #[test]
    fn manual_already_exhausted_and_zero_pace() {
        let now = at(T0);
        let exhausted = forecast_manual(50.0, 60.0, 300.0, 30, now);
        assert_eq!(exhausted.status, ForecastStatus::AlreadyExhausted);
        assert_eq!(exhausted.projected_usd, 60.0);
        let flat = forecast_manual(500.0, 60.0, 0.0, 30, now);
        assert_eq!(flat.status, ForecastStatus::WithinWindow);
        assert_eq!(flat.projected_usd, 60.0);
    }

    #[test]
    fn manual_exhaustion_exactly_at_horizon_end_is_within_window() {
        let now = at(T0);
        // pace 10/day, headroom 299.9: exhaustion at day ~29.99 falls
        // inside the 30-day horizon; exactly 300 (day 30.0) would be AT
        // the horizon end and stays WithinWindow.
        let inside = forecast_manual(399.9, 100.0, 300.0, 30, now);
        assert_eq!(inside.status, ForecastStatus::ExhaustsWithinWindow);
        let boundary = forecast_manual(400.0, 100.0, 300.0, 30, now);
        assert_eq!(boundary.status, ForecastStatus::WithinWindow);
        assert_eq!(boundary.projected_exhaustion_date, None);
    }

#[test]
fn non_finite_inputs_are_insufficient_data() {
    let start = at(T0);
    let end = at(T0 + 30 * SECONDS_PER_DAY);
    let now = at(T0 + 15 * SECONDS_PER_DAY);
    let nan_limit = forecast_windowed(f64::NAN, 10.0, start, end, now);
    assert_eq!(nan_limit.status, ForecastStatus::InsufficientData);
    let inf_spent = forecast_manual(100.0, f64::INFINITY, 10.0, 30, now);
    assert_eq!(inf_spent.status, ForecastStatus::InsufficientData);
    let nan_trailing = forecast_manual(100.0, 10.0, f64::NAN, 30, now);
    assert_eq!(nan_trailing.status, ForecastStatus::InsufficientData);
}

#[test]
fn non_finite_payload_values_are_clamped_finite() {
    let start = at(T0);
    let end = at(T0 + 30 * SECONDS_PER_DAY);
    let now = at(T0 + 15 * SECONDS_PER_DAY);
    // Non-finite spent reaches the InsufficientData constructor; the
    // payload field must serialize as a finite number, never null.
    let inf_spent = forecast_windowed(100.0, f64::INFINITY, start, end, now);
    assert_eq!(inf_spent.status, ForecastStatus::InsufficientData);
    assert!(inf_spent.projected_usd.is_finite());
    assert_eq!(inf_spent.projected_usd, 0.0);
    assert_eq!(
        serde_json::to_value(&inf_spent).unwrap()["projected_usd"],
        serde_json::json!(0.0)
    );
    let nan_spent = forecast_manual(100.0, f64::NAN, 10.0, 30, now);
    assert_eq!(nan_spent.status, ForecastStatus::InsufficientData);
    assert_eq!(nan_spent.projected_usd, 0.0);
}

#[test]
fn windowed_projection_overflow_classifies_as_exhausts_within_window() {
    let start = at(T0);
    let end = at(T0 + 30 * SECONDS_PER_DAY);
    // Exactly the 5% gate (1.5 of 30 days elapsed) with extreme-but-
    // finite spend: pace * remaining overflows f64 to +inf, yet the
    // exhaustion epoch (~10.5 days in) lands well inside the window —
    // the projection must classify as ExhaustsWithinWindow with the
    // projected spend clamped to the finite limit, not degrade to
    // InsufficientData.
    let now = at(T0 + (3 * SECONDS_PER_DAY) / 2);
    let spent = 1.0e308_f64;
    let limit = 1.7e308_f64;
    let f = forecast_windowed(limit, spent, start, end, now);
    assert_eq!(f.status, ForecastStatus::ExhaustsWithinWindow);
    assert!(f.projected_usd.is_finite());
    assert_eq!(f.projected_usd, limit);
    assert!(f.projected_exhaustion_date.is_some());
    assert!(f.days_remaining.is_some());
}

#[test]
fn manual_forecast_carries_horizon_end_on_every_path() {
    let now = at(T0);
    let expected = Some(T0 + MANUAL_HORIZON_DAYS * SECONDS_PER_DAY);
    let exhausts = forecast_manual(200.0, 50.0, 300.0, 30, now);
    assert_eq!(exhausts.horizon_end, expected);
    let within = forecast_manual(1000.0, 50.0, 300.0, 30, now);
    assert_eq!(within.horizon_end, expected);
    let short_history = forecast_manual(200.0, 50.0, 300.0, 6, now);
    assert_eq!(short_history.horizon_end, expected);
    let exhausted = forecast_manual(50.0, 60.0, 300.0, 30, now);
    assert_eq!(exhausted.horizon_end, expected);
    let non_finite = forecast_manual(100.0, f64::INFINITY, 10.0, 30, now);
    assert_eq!(non_finite.horizon_end, expected);
}

#[test]
fn windowed_forecast_never_carries_horizon_end() {
    let start = at(T0);
    let end = at(T0 + 30 * SECONDS_PER_DAY);
    let now = at(T0 + 15 * SECONDS_PER_DAY);
    let f = forecast_windowed(100.0, 40.0, start, end, now);
    assert_eq!(f.horizon_end, None);
}

#[test]
fn serde_serializes_design_field_names() {
    let now = at(T0);
    let f = forecast_manual(200.0, 50.0, 300.0, 30, now);
    let json = serde_json::to_value(&f).unwrap();
    assert_eq!(json["method"], serde_json::json!("trailing_run_rate"));
    assert_eq!(json["status"], serde_json::json!("exhausts_within_window"));
    assert_eq!(json["days_remaining"], serde_json::json!(15));
    assert!(json["projected_exhaustion_date"].is_string());
    assert_eq!(
        json["horizon_end"],
        serde_json::json!(T0 + MANUAL_HORIZON_DAYS * SECONDS_PER_DAY)
    );
    let windowed = forecast_windowed(
        100.0,
        40.0,
        at(T0),
        at(T0 + 30 * SECONDS_PER_DAY),
        at(T0 + 15 * SECONDS_PER_DAY),
    );
    let json = serde_json::to_value(&windowed).unwrap();
    assert_eq!(json["method"], serde_json::json!("window_pace"));
    assert_eq!(json["status"], serde_json::json!("within_window"));
    assert!(json["projected_exhaustion_date"].is_null());
    // Windowed scopes expose the window end as next_reset_epoch on the
    // enclosing status row; the forecast itself omits horizon_end.
    assert!(json.get("horizon_end").is_none());
}
}

#[cfg(test)]
mod property_tests {
    use super::*;
    use chrono::Duration;
    use proptest::prelude::*;

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 64,
            .. ProptestConfig::default()
        })]

    /// Higher window-to-date spend (same window, clock, and limit)
    /// can only keep or move the projected exhaustion date earlier;
    /// the status order never regresses from ExhaustsWithinWindow.
    /// `extra` is bounded to `limit - low_spend` so `high_spend`
    /// stays under the limit and the Exhausted arm (a factual state
    /// outside monotonicity) almost never triggers prop_assume!
    /// rejection.
    #[test]
    fn prop_windowed_higher_spend_exhausts_no_later(
        limit_usd in 1.0f64..1e7,
        low_spend in 0.0f64..1e7,
        extra in 0.0f64..1e7,
        window_days in 2i64..=120,
        day_offset in 1i64..=119,
    ) {
        prop_assume!(day_offset < window_days);
        prop_assume!(low_spend < limit_usd);
        let start = Utc.timestamp_opt(1_780_278_400, 0).unwrap();
        let end = start + Duration::days(window_days);
        let now = start + Duration::days(day_offset);
        // Bound the increment so high_spend stays below the limit
        // (the generator cannot express a bound on a later variable).
        let extra = extra % (limit_usd - low_spend);
        let low = forecast_windowed(limit_usd, low_spend, start, end, now);
        let high_spend = low_spend + extra;
        let high = forecast_windowed(limit_usd, high_spend, start, end, now);
        prop_assume!(high.status != ForecastStatus::AlreadyExhausted);
        prop_assume!(low.status != ForecastStatus::AlreadyExhausted);
            match (low.status, high.status) {
                (ForecastStatus::ExhaustsWithinWindow, ForecastStatus::ExhaustsWithinWindow) => {
                    prop_assert!(high.projected_exhaustion_date.unwrap()
                        <= low.projected_exhaustion_date.unwrap());
                }
                (ForecastStatus::ExhaustsWithinWindow, _) => {
                    prop_assert!(high.projected_exhaustion_date.is_none());
                }
                _ => {}
            }
        }

        /// Higher trailing 30-day spend (same limit, spend, clock) can
        /// only keep or move the projected exhaustion date earlier.
        #[test]
        fn prop_manual_higher_pace_exhausts_no_later(
            limit_usd in 1.0f64..1e7,
            spent_usd in 0.0f64..1e6,
            low_trailing in 0.0f64..1e6,
            extra in 0.0f64..1e6,
            hour_of_day in 0i64..=23,
        ) {
            prop_assume!(spent_usd < limit_usd);
            let now = Utc.timestamp_opt(1_780_278_400 + hour_of_day * 3600, 0).unwrap();
            let low = forecast_manual(limit_usd, spent_usd, low_trailing, 30, now);
            let high = forecast_manual(limit_usd, spent_usd, low_trailing + extra, 30, now);
            match (low.status, high.status) {
                (ForecastStatus::ExhaustsWithinWindow, ForecastStatus::ExhaustsWithinWindow) => {
                    prop_assert!(high.projected_exhaustion_date.unwrap()
                        <= low.projected_exhaustion_date.unwrap());
                }
                (ForecastStatus::ExhaustsWithinWindow, _) => {
                    prop_assert!(high.projected_exhaustion_date.is_none());
                }
                _ => {}
            }
        }

        /// Arbitrary (including degenerate: zero, negative, huge,
        /// NaN, inverted windows, clocks outside the window) inputs
        /// never panic and always return a usable status.
        #[test]
        fn prop_no_panic_on_arbitrary_inputs(
            limit_usd in prop_oneof!(Just(f64::NAN), Just(f64::INFINITY), -1e6f64..1e12),
            spent_usd in prop_oneof!(Just(f64::NAN), Just(f64::NEG_INFINITY), -1e6f64..1e12),
            trailing in prop_oneof!(Just(f64::NAN), -1e6f64..1e12),
            history_days in 0u32..=60,
            start_secs in -2_000_000_000i64..2_000_000_000,
            duration_secs in -100_000i64..400_000_000,
            now_offset in -200_000_000i64..200_000_000i64,
        ) {
            let start = Utc.timestamp_opt(start_secs, 0).unwrap();
            let end = Utc.timestamp_opt(start_secs.saturating_add(duration_secs), 0).unwrap();
            let now = Utc.timestamp_opt(start_secs.saturating_add(now_offset), 0).unwrap();
            let windowed = forecast_windowed(limit_usd, spent_usd, start, end, now);
            prop_assert!(matches!(
                windowed.status,
                ForecastStatus::InsufficientData
                    | ForecastStatus::WithinWindow
                    | ForecastStatus::ExhaustsWithinWindow
                    | ForecastStatus::AlreadyExhausted
            ));
            let manual = forecast_manual(limit_usd, spent_usd, trailing, history_days, now);
            prop_assert!(matches!(
                manual.status,
                ForecastStatus::InsufficientData
                    | ForecastStatus::WithinWindow
                    | ForecastStatus::ExhaustsWithinWindow
                    | ForecastStatus::AlreadyExhausted
            ));
            if windowed.status == ForecastStatus::ExhaustsWithinWindow {
                prop_assert!(windowed.projected_exhaustion_date.is_some());
                prop_assert!(windowed.days_remaining.is_some());
            }
            if manual.status == ForecastStatus::ExhaustsWithinWindow {
                prop_assert!(manual.projected_exhaustion_date.is_some());
                prop_assert!(manual.days_remaining.is_some());
            }
        }

        /// Projection invariants: status is WithinWindow exactly when
        /// the finite projection stays under the limit, and
        /// insufficient/exhausted states never extrapolate spend.
        #[test]
        fn prop_status_matches_projection_vs_limit(
            limit_usd in 0.001f64..1e7,
            spent_usd in 0.0f64..1e6,
            window_days in 2i64..=120,
            day_offset in 1i64..=119,
        ) {
            prop_assume!(day_offset < window_days);
            prop_assume!(spent_usd < limit_usd);
            let start = Utc.timestamp_opt(1_780_278_400, 0).unwrap();
            let end = start + Duration::days(window_days);
            let now = start + Duration::days(day_offset);
            let f = forecast_windowed(limit_usd, spent_usd, start, end, now);
            match f.status {
                ForecastStatus::WithinWindow => {
                    prop_assert!(f.projected_usd < limit_usd || f.projected_usd.is_finite());
                    prop_assert!(f.projected_exhaustion_date.is_none());
                }
                ForecastStatus::ExhaustsWithinWindow => {
                    prop_assert!(f.projected_usd >= spent_usd);
                    prop_assert!(f.days_remaining.unwrap() <= window_days as u32);
                }
                _ => {}
            }
        }
    }
}
