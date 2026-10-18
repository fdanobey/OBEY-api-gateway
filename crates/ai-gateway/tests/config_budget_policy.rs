//! Config contract tests for `providers.<name>.budget.reset_policy`
//! (spec: persistent-budgets-rolling-rate, Task 1).
//!
//! Zero-config compatibility: YAML with only `limit_usd` must keep parsing
//! with the policy defaulting to `manual`; unknown policy strings must be
//! rejected loudly by serde.

use ai_gateway::config::{BudgetResetPolicy, ProviderBudgetConfig};
use ai_gateway::virtual_keys::models::BudgetWindow;

#[test]
fn budget_with_only_limit_usd_defaults_reset_policy_to_manual() {
    let parsed: ProviderBudgetConfig =
        serde_yaml::from_str("limit_usd: 25.0").expect("legacy budget YAML must parse");
    assert_eq!(parsed.limit_usd, 25.0);
    assert_eq!(parsed.reset_policy, None);
    assert_eq!(
        parsed.reset_policy.unwrap_or_default(),
        BudgetResetPolicy::Manual,
        "absent reset_policy must behave as manual"
    );
}

#[test]
fn budget_reset_policy_round_trips_all_variants() {
    for (yaml, expected) in [
        ("daily", BudgetResetPolicy::Daily),
        ("weekly", BudgetResetPolicy::Weekly),
        ("monthly", BudgetResetPolicy::Monthly),
        ("manual", BudgetResetPolicy::Manual),
    ] {
        let parsed: ProviderBudgetConfig =
            serde_yaml::from_str(&format!("limit_usd: 1.0\nreset_policy: {yaml}\n"))
                .expect("known policy strings must parse");
        assert_eq!(parsed.reset_policy, Some(expected), "mismatch for {yaml}");
    }
}

#[test]
fn invalid_budget_reset_policy_string_is_rejected() {
    let result: Result<ProviderBudgetConfig, _> =
        serde_yaml::from_str("limit_usd: 1.0\nreset_policy: fortnightly\n");
    let err = result.expect_err("unknown policy strings must be rejected by serde");
    assert!(
        err.to_string().to_lowercase().contains("fortnightly"),
        "error should name the offending value, got: {err}"
    );
}

#[test]
fn budget_reset_policy_maps_to_budget_window_semantics() {
    use ai_gateway::config::BudgetResetPolicy as P;
    let window: Option<BudgetWindow> = Option::<BudgetWindow>::from(P::Daily);
    assert_eq!(window, Some(BudgetWindow::Daily));
    let window: Option<BudgetWindow> = P::Weekly.into();
    assert_eq!(window, Some(BudgetWindow::Weekly));
    let window: Option<BudgetWindow> = P::Monthly.into();
    assert_eq!(window, Some(BudgetWindow::Monthly));
    let window: Option<BudgetWindow> = P::Manual.into();
    assert_eq!(window, None, "manual must map to no periodic window");
}
