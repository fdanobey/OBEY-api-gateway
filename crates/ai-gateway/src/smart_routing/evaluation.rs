//! Offline routing evaluation harness for quality-cost frontier analysis.
//!
//! Provides deterministic library evaluation of routing policies against
//! knowledge, math, code, conversation, tool-use, structured-output,
//! long-context, and operator-preference cases without invoking model APIs.
//!
//! Requirements: 16.1, 17.1, 22.1, 23.5

use std::collections::BTreeMap;
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::tier::SmartRoutingTier;

#[allow(dead_code)] // kept for completeness of the validation API
fn validate_finite(value: f64, field: &str) -> Result<(), EvaluationInputError> {
    if !value.is_finite() {
        return Err(EvaluationInputError::NonFiniteValue {
            field: field.to_string(),
        });
    }
    Ok(())
}

fn validate_non_negative(value: f64, field: &str) -> Result<(), EvaluationInputError> {
    if !value.is_finite() || value < 0.0 {
        return Err(EvaluationInputError::NonFiniteValue {
            field: field.to_string(),
        });
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum EvaluationDomain {
    Knowledge,
    Math,
    Code,
    Conversation,
    ToolUse,
    StructuredOutput,
    LongContext,
    OperatorPreference,
    Custom(String),
}

impl EvaluationDomain {
    fn validate(&self) -> Result<(), EvaluationInputError> {
        match self {
            Self::Custom(name) if name.len() > 128 || name.is_empty() => {
                Err(EvaluationInputError::InvalidDomainName)
            }
            _ => Ok(()),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExpectedOutcome {
    pub task_type: String,
    pub tier: SmartRoutingTier,
    pub quality_floor: f64,
    pub cost_ceiling_usd: Option<f64>,
    pub escalation_expected: bool,
}

impl ExpectedOutcome {
    fn validate(&self) -> Result<(), EvaluationInputError> {
        validate_non_negative(self.quality_floor, "quality_floor")?;
        if self.quality_floor > 1.0 {
            return Err(EvaluationInputError::NonFiniteValue {
                field: "quality_floor".to_string(),
            });
        }
        if let Some(cost) = self.cost_ceiling_usd {
            validate_non_negative(cost, "cost_ceiling_usd")?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProviderScenario {
    pub provider: String,
    pub model: String,
    pub success: bool,
    pub simulated_latency_ms: f64,
    pub simulated_cost_usd: f64,
}

impl ProviderScenario {
    fn validate(&self) -> Result<(), EvaluationInputError> {
        validate_non_negative(self.simulated_latency_ms, "simulated_latency_ms")?;
        validate_non_negative(self.simulated_cost_usd, "simulated_cost_usd")?;
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationCase {
    pub id: String,
    pub domain: EvaluationDomain,
    pub messages_json: serde_json::Value,
    pub expected: ExpectedOutcome,
    pub providers: Vec<ProviderScenario>,
}

impl EvaluationCase {
    fn validate(&self) -> Result<(), EvaluationInputError> {
        if self.id.is_empty() || self.id.len() > 256 {
            return Err(EvaluationInputError::InvalidCaseId);
        }
        self.domain.validate()?;
        self.expected.validate()?;
        for provider in &self.providers {
            provider.validate()?;
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum EvaluationInputError {
    InvalidUtf8,
    InvalidJson,
    InvalidCaseId,
    InvalidDomainName,
    NonFiniteValue { field: String },
    FilePathOutsideRoot,
    SymlinkRejected,
}

impl std::fmt::Display for EvaluationInputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidUtf8 => f.write_str("evaluation file must be UTF-8 encoded"),
            Self::InvalidJson => f.write_str("evaluation file must contain valid JSON"),
            Self::InvalidCaseId => {
                f.write_str("case id must be non-empty and at most 256 characters")
            }
            Self::InvalidDomainName => {
                f.write_str("custom domain names must be non-empty and at most 128 characters")
            }
            Self::NonFiniteValue { field } => {
                write!(f, "{field} must be finite and within bounds")
            }
            Self::FilePathOutsideRoot => {
                f.write_str("evaluation paths must remain within the provided root")
            }
            Self::SymlinkRejected => {
                f.write_str("symbolic links are not allowed in evaluation paths")
            }
        }
    }
}

impl std::error::Error for EvaluationInputError {}

#[derive(Debug, Clone, Default, Serialize)]
pub struct EvaluationMetrics {
    pub cases_evaluated: u64,
    pub strong_call_rate: f64,
    pub quality_retention: f64,
    pub realized_cost_usd: f64,
    pub escalation_rate: f64,
    pub calibration_error: f64,
    pub average_latency_ms: f64,
    pub domain_breakdown: BTreeMap<String, DomainMetrics>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct DomainMetrics {
    pub n: u64,
    pub quality_retention: f64,
    pub cost_usd: f64,
    pub escalation_rate: f64,
}

impl EvaluationMetrics {
    fn finite_or_zero(&mut self) {
        self.strong_call_rate = if self.strong_call_rate.is_finite() {
            self.strong_call_rate
        } else {
            0.0
        };
        self.quality_retention = if self.quality_retention.is_finite() {
            self.quality_retention
        } else {
            0.0
        };
        self.realized_cost_usd = if self.realized_cost_usd.is_finite() {
            self.realized_cost_usd
        } else {
            0.0
        };
        self.escalation_rate = if self.escalation_rate.is_finite() {
            self.escalation_rate
        } else {
            0.0
        };
        self.calibration_error = if self.calibration_error.is_finite() {
            self.calibration_error
        } else {
            0.0
        };
        self.average_latency_ms = if self.average_latency_ms.is_finite() {
            self.average_latency_ms
        } else {
            0.0
        };
    }
}

pub trait EvaluationRouter {
    fn route(&self, case: &EvaluationCase) -> EvaluationOutcome;
}

#[derive(Debug, Clone, PartialEq)]
pub struct EvaluationOutcome {
    pub selected_tier: SmartRoutingTier,
    pub selected_provider: Option<String>,
    pub selected_model: Option<String>,
    pub quality_score: f64,
    pub cost_usd: f64,
    pub latency_ms: f64,
    pub escalated: bool,
}

pub struct EvaluationHarness;

impl EvaluationHarness {
    pub fn load_cases(
        root: &Path,
        path: &Path,
    ) -> Result<Vec<EvaluationCase>, EvaluationInputError> {
        let canonical_root = root
            .canonicalize()
            .map_err(|_| EvaluationInputError::FilePathOutsideRoot)?;
        if std::fs::symlink_metadata(path)
            .map_err(|_| EvaluationInputError::FilePathOutsideRoot)?
            .file_type()
            .is_symlink()
        {
            return Err(EvaluationInputError::SymlinkRejected);
        }
        let canonical_path = path
            .canonicalize()
            .map_err(|_| EvaluationInputError::FilePathOutsideRoot)?;
        if !canonical_path.starts_with(&canonical_root) {
            return Err(EvaluationInputError::FilePathOutsideRoot);
        }
        let text = std::fs::read_to_string(&canonical_path)
            .map_err(|_| EvaluationInputError::InvalidUtf8)?;
        let cases: Vec<EvaluationCase> =
            serde_json::from_str(&text).map_err(|_| EvaluationInputError::InvalidJson)?;
        for case in &cases {
            case.validate()?;
        }
        Ok(cases)
    }

    pub fn evaluate(cases: &[EvaluationCase], router: &dyn EvaluationRouter) -> EvaluationMetrics {
        let mut metrics = EvaluationMetrics::default();
        let mut total_latency = 0.0_f64;
        let mut quality_sum = 0.0_f64;
        let mut calibration_sum = 0.0_f64;
        let mut strong_calls = 0u64;

        for case in cases {
            let outcome = router.route(case);
            let domain = match &case.domain {
                EvaluationDomain::Knowledge => "knowledge",
                EvaluationDomain::Math => "math",
                EvaluationDomain::Code => "code",
                EvaluationDomain::Conversation => "conversation",
                EvaluationDomain::ToolUse => "tool_use",
                EvaluationDomain::StructuredOutput => "structured_output",
                EvaluationDomain::LongContext => "long_context",
                EvaluationDomain::OperatorPreference => "operator_preference",
                EvaluationDomain::Custom(name) => name.as_str(),
            };
            let entry = metrics
                .domain_breakdown
                .entry(domain.to_string())
                .or_default();
            entry.n += 1;
            metrics.cases_evaluated += 1;
            if outcome.selected_tier == SmartRoutingTier::Powerful {
                strong_calls += 1;
            }
            quality_sum += outcome.quality_score;
            metrics.realized_cost_usd += outcome.cost_usd;
            entry.cost_usd += outcome.cost_usd;
            total_latency += outcome.latency_ms;
            if outcome.escalated {
                metrics.escalation_rate += 1.0;
                entry.escalation_rate += 1.0;
            }
            calibration_sum += (outcome.quality_score - case.expected.quality_floor).abs();
            entry.quality_retention += if outcome.quality_score >= case.expected.quality_floor {
                1.0
            } else {
                0.0
            };
        }

        let n = metrics.cases_evaluated.max(1) as f64;
        metrics.strong_call_rate = strong_calls as f64 / n;
        metrics.quality_retention = quality_sum / n;
        metrics.escalation_rate /= n;
        metrics.calibration_error = calibration_sum / n;
        metrics.average_latency_ms = total_latency / n;

        for domain_metrics in metrics.domain_breakdown.values_mut() {
            let dn = domain_metrics.n.max(1) as f64;
            domain_metrics.quality_retention /= dn;
            domain_metrics.escalation_rate /= dn;
        }

        metrics.finite_or_zero();
        metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    struct StaticRouter {
        tier: SmartRoutingTier,
        quality: f64,
        cost: f64,
        latency: f64,
    }

    impl EvaluationRouter for StaticRouter {
        fn route(&self, _case: &EvaluationCase) -> EvaluationOutcome {
            EvaluationOutcome {
                selected_tier: self.tier,
                selected_provider: Some("static-provider".to_string()),
                selected_model: Some("static-model".to_string()),
                quality_score: self.quality,
                cost_usd: self.cost,
                latency_ms: self.latency,
                escalated: false,
            }
        }
    }

    fn example_case(id: &str, quality_floor: f64) -> EvaluationCase {
        EvaluationCase {
            id: id.to_string(),
            domain: EvaluationDomain::Knowledge,
            messages_json: serde_json::json!([{"role":"user","content":"test"}]),
            expected: ExpectedOutcome {
                task_type: "knowledge".to_string(),
                tier: SmartRoutingTier::Balanced,
                quality_floor,
                cost_ceiling_usd: None,
                escalation_expected: false,
            },
            providers: vec![ProviderScenario {
                provider: "static-provider".to_string(),
                model: "static-model".to_string(),
                success: true,
                simulated_latency_ms: 100.0,
                simulated_cost_usd: 0.001,
            }],
        }
    }

    #[test]
    fn load_cases_rejects_paths_outside_root() {
        let dir = tempdir().unwrap();
        let outside = tempdir().unwrap();
        let file_path = outside.path().join("cases.json");
        std::fs::write(&file_path, "[]").unwrap();
        assert!(matches!(
            EvaluationHarness::load_cases(dir.path(), &file_path),
            Err(EvaluationInputError::FilePathOutsideRoot)
        ));
    }

    #[test]
    fn load_cases_rejects_symlinks() {
        let dir = tempdir().unwrap();
        let target = dir.path().join("real.json");
        std::fs::write(&target, "[]").unwrap();
        let link = dir.path().join("link.json");
        let created = {
            #[cfg(unix)]
            {
                std::os::unix::fs::symlink(&target, &link).is_ok()
            }
            #[cfg(windows)]
            {
                std::os::windows::fs::symlink_file(&target, &link).is_ok()
            }
        };
        if !created {
            eprintln!("symlink creation not permitted on this platform; skipping");
            return;
        }
        let result = EvaluationHarness::load_cases(dir.path(), &link);
        assert!(matches!(result, Err(EvaluationInputError::SymlinkRejected)));
    }

    #[test]
    fn load_cases_rejects_non_utf8() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("cases.json");
        std::fs::write(&file, b"\xFF\xFE").unwrap();
        assert!(matches!(
            EvaluationHarness::load_cases(dir.path(), &file),
            Err(EvaluationInputError::InvalidUtf8)
        ));
    }

    #[test]
    fn evaluate_aggregates_metrics_correctly() {
        let cases = vec![example_case("case-1", 0.5), example_case("case-2", 0.8)];
        let router = StaticRouter {
            tier: SmartRoutingTier::Balanced,
            quality: 0.6,
            cost: 0.001,
            latency: 50.0,
        };
        let metrics = EvaluationHarness::evaluate(&cases, &router);
        assert_eq!(metrics.cases_evaluated, 2);
        assert!((metrics.quality_retention - 0.6).abs() < 1e-9);
        assert!((metrics.realized_cost_usd - 0.002).abs() < 1e-9);
        assert!((metrics.average_latency_ms - 50.0).abs() < 1e-9);
        assert_eq!(metrics.strong_call_rate, 0.0);
    }

    #[test]
    fn metrics_fields_are_finite_and_bounded() {
        let cases = vec![example_case("case", 0.2)];
        let router = StaticRouter {
            tier: SmartRoutingTier::Fast,
            quality: f64::INFINITY,
            cost: f64::NAN,
            latency: -10.0,
        };
        let metrics = EvaluationHarness::evaluate(&cases, &router);
        assert!(metrics.quality_retention.is_finite());
        assert!(metrics.realized_cost_usd.is_finite());
        assert!(metrics.average_latency_ms.is_finite());
        assert!(metrics.calibration_error.is_finite());
    }

    #[test]
    fn validation_rejects_infinite_quality_floor() {
        let case = example_case("bad", f64::INFINITY);
        assert!(matches!(
            case.validate(),
            Err(EvaluationInputError::NonFiniteValue { .. })
        ));
    }
}
