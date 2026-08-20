pub mod config;
pub mod metrics;
pub mod retry;
pub mod validator;

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use futures::future::join_all;

use crate::config::Config;
use crate::models::openai::{OpenAIRequest, OpenAIResponse};
use config::{EffectiveConfig, StructuredOutputConfig, StructuredOutputOverride};
use metrics::StructuredOutputMetrics;
use validator::{
    ChoiceValidationOutcome, ChoiceValidationResult, MalformedResponseFormat, SchemaCompileError,
    SchemaContext, SchemaContextExtraction, SchemaViolation,
};

#[derive(Clone)]
pub struct StructuredOutputEngine {
    global: StructuredOutputConfig,
    groups: HashMap<String, GroupPolicy>,
    metrics: Arc<StructuredOutputMetrics>,
}

#[derive(Clone, Default)]
struct GroupPolicy {
    config_override: Option<StructuredOutputOverride>,
    provider_passthrough: HashMap<(String, String), bool>,
}

#[derive(Clone)]
pub enum ValidationDecision {
    NotApplicable,
    Skipped(ValidationSkipReason),
    Validate(SchemaContext),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ValidationSkipReason {
    Disabled,
    Passthrough,
    Malformed(MalformedResponseFormat),
    CompileFailed(SchemaCompileError),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StructuredOutputOutcome {
    NotApplicable,
    Pass,
    Fail,
    Skipped,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ResponseValidationOutcome {
    pub outcome: StructuredOutputOutcome,
    pub choices: Vec<ChoiceValidationOutcome>,
    pub latency_ms: f64,
}

impl StructuredOutputEngine {
    /// Builds an engine only when structured output is explicitly configured.
    pub fn from_config(config: &Config) -> Option<Self> {
        Self::from_config_with_metrics(config, Arc::new(StructuredOutputMetrics::new()))
    }

    pub fn from_config_with_metrics(
        config: &Config,
        metrics: Arc<StructuredOutputMetrics>,
    ) -> Option<Self> {
        let global = config.structured_output.clone()?;
        let groups = config
            .model_groups
            .iter()
            .map(|group| {
                let provider_passthrough = group
                    .models
                    .iter()
                    .filter_map(|provider_model| {
                        provider_model
                            .structured_output_passthrough
                            .map(|passthrough| {
                                (
                                    (
                                        provider_model.provider.clone(),
                                        provider_model.model.clone(),
                                    ),
                                    passthrough,
                                )
                            })
                    })
                    .collect();

                (
                    group.name.clone(),
                    GroupPolicy {
                        config_override: group.structured_output.clone(),
                        provider_passthrough,
                    },
                )
            })
            .collect();

        Some(Self {
            global,
            groups,
            metrics,
        })
    }

    pub fn metrics(&self) -> Arc<StructuredOutputMetrics> {
        Arc::clone(&self.metrics)
    }

    /// Resolves global values, group overrides, and passthrough precedence.
    pub fn effective_config(
        &self,
        model_group: &str,
        provider: &str,
        model: &str,
    ) -> EffectiveConfig {
        let group = self.groups.get(model_group);
        let merged = group
            .and_then(|policy| policy.config_override.as_ref())
            .map_or_else(|| self.global.clone(), |value| self.global.merge(value));

        let passthrough = group
            .and_then(|policy| {
                policy
                    .provider_passthrough
                    .get(&(provider.to_owned(), model.to_owned()))
                    .copied()
            })
            .or_else(|| {
                group
                    .and_then(|policy| policy.config_override.as_ref())
                    .and_then(|value| value.passthrough_providers.as_ref())
                    .map(|providers| provider_is_listed(providers, provider))
            })
            .unwrap_or_else(|| provider_is_listed(&self.global.passthrough_providers, provider));

        EffectiveConfig {
            enabled: merged.enabled,
            max_retries: merged.max_retries,
            retry_temperature: merged.retry_temperature,
            passthrough,
        }
    }

    /// Decides whether a request should be validated and compiles its schema.
    ///
    /// Disabled and passthrough requests avoid schema compilation. Applicable
    /// skips are recorded, while requests without `response_format` remain
    /// entirely outside structured-output accounting.
    pub fn should_validate(
        &self,
        request: &OpenAIRequest,
        model_group: &str,
        provider: &str,
        model: &str,
    ) -> ValidationDecision {
        if !request.extra.contains_key("response_format") {
            return ValidationDecision::NotApplicable;
        }

        let effective = self.effective_config(model_group, provider, model);
        let decision = if !effective.enabled {
            ValidationDecision::Skipped(ValidationSkipReason::Disabled)
        } else if effective.passthrough {
            ValidationDecision::Skipped(ValidationSkipReason::Passthrough)
        } else {
            match validator::extract_schema_context(request) {
                SchemaContextExtraction::NotApplicable => ValidationDecision::NotApplicable,
                SchemaContextExtraction::Malformed(error) => {
                    ValidationDecision::Skipped(ValidationSkipReason::Malformed(error))
                }
                SchemaContextExtraction::CompileFailed(error) => {
                    ValidationDecision::Skipped(ValidationSkipReason::CompileFailed(error))
                }
                SchemaContextExtraction::Ready(context) => ValidationDecision::Validate(context),
            }
        };

        if matches!(decision, ValidationDecision::Skipped(_)) {
            self.metrics
                .record_structured_output_validation(provider, model, "skip");
        }

        decision
    }

    /// Validates all response choices concurrently through the validator's
    /// bounded asynchronous API and reduces them to one request-level outcome.
    pub async fn validate_response(
        &self,
        context: &SchemaContext,
        response: &OpenAIResponse,
        provider: &str,
        model: &str,
    ) -> ResponseValidationOutcome {
        let started = Instant::now();
        let choices = join_all(response.choices.iter().map(|choice| {
            validator::validate_response_async(
                Arc::clone(&context.compiled),
                choice.message.content_as_text(),
            )
        }))
        .await;

        let outcome = aggregate_outcome(&choices);
        self.metrics.record_structured_output_validation(
            provider,
            model,
            validation_metric_status(outcome),
        );
        let latency_ms = started.elapsed().as_secs_f64() * 1_000.0;

        ResponseValidationOutcome {
            outcome,
            choices,
            latency_ms,
        }
    }

    /// Delegates corrective retry construction to the retry module.
    #[allow(clippy::too_many_arguments)]
    pub fn build_retry_request(
        &self,
        original: &OpenAIRequest,
        context: &SchemaContext,
        errors: &[SchemaViolation],
        previous_output: &str,
        effective_config: &EffectiveConfig,
        original_was_streaming: bool,
        context_window_token_limit: usize,
        current_original_token_estimate: usize,
    ) -> OpenAIRequest {
        retry::build_retry_request(
            original,
            &context.raw_schema,
            context.schema_char_len,
            errors,
            previous_output,
            effective_config.retry_temperature,
            original_was_streaming,
            context_window_token_limit,
            current_original_token_estimate,
        )
    }

    pub fn is_cache_eligible(outcome: StructuredOutputOutcome) -> bool {
        matches!(
            outcome,
            StructuredOutputOutcome::NotApplicable | StructuredOutputOutcome::Pass
        )
    }
}

fn provider_is_listed(providers: &[String], provider: &str) -> bool {
    providers.iter().any(|candidate| candidate == provider)
}

fn aggregate_outcome(choices: &[ChoiceValidationOutcome]) -> StructuredOutputOutcome {
    if choices.iter().any(|choice| {
        matches!(
            choice.result,
            ChoiceValidationResult::JsonParseError { .. }
                | ChoiceValidationResult::SchemaViolations(_)
        )
    }) {
        StructuredOutputOutcome::Fail
    } else if choices.is_empty()
        || choices
            .iter()
            .any(|choice| choice.result == ChoiceValidationResult::Skipped)
    {
        StructuredOutputOutcome::Skipped
    } else {
        StructuredOutputOutcome::Pass
    }
}

fn validation_metric_status(outcome: StructuredOutputOutcome) -> &'static str {
    match outcome {
        StructuredOutputOutcome::Pass => "pass",
        StructuredOutputOutcome::Fail => "fail",
        StructuredOutputOutcome::NotApplicable | StructuredOutputOutcome::Skipped => "skip",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::ProviderModel;
    use crate::models::openai::{Choice, Message, Usage};
    use proptest::prelude::*;
    use serde_json::{json, Map, Value};

    fn config_with_structured_output(structured_output: &str, model_groups: &str) -> Config {
        serde_yaml::from_str(&format!(
            r#"
server:
  host: 127.0.0.1
  port: 8080
providers:
  - name: provider-config
    type: openai
model_groups:
{model_groups}
structured_output:
{structured_output}
"#
        ))
        .unwrap()
    }

    fn request(response_format: Option<Value>) -> OpenAIRequest {
        let mut extra = Map::new();
        if let Some(response_format) = response_format {
            extra.insert("response_format".to_owned(), response_format);
        }
        OpenAIRequest {
            model: "client-model".to_owned(),
            messages: Vec::new(),
            stream: false,
            temperature: None,
            max_tokens: None,
            extra,
        }
    }

    fn schema_request() -> OpenAIRequest {
        request(Some(json!({
            "type": "json_schema",
            "json_schema": {
                "schema": {
                    "type": "object",
                    "properties": {"id": {"type": "integer"}},
                    "required": ["id"]
                }
            }
        })))
    }

    fn response(contents: Vec<Value>) -> OpenAIResponse {
        OpenAIResponse {
            id: String::new(),
            object: String::new(),
            created: 0,
            model: "upstream-model".to_owned(),
            choices: contents
                .into_iter()
                .enumerate()
                .map(|(index, content)| Choice {
                    index: index as u32,
                    message: Message {
                        role: "assistant".to_owned(),
                        content,
                        extra: Map::new(),
                    },
                    finish_reason: Some("stop".to_owned()),
                    extra: Map::new(),
                })
                .collect(),
            usage: Usage::default(),
            extra: Map::new(),
        }
    }

    fn active_context(engine: &StructuredOutputEngine) -> SchemaContext {
        match engine.should_validate(&schema_request(), "group", "active", "model") {
            ValidationDecision::Validate(context) => context,
            _ => panic!("expected active schema context"),
        }
    }

    #[derive(Debug, Clone)]
    enum GeneratedChoice {
        Pass(i64),
        ParseError,
        SchemaViolation(String),
        Skipped,
    }

    impl GeneratedChoice {
        fn content(&self) -> Value {
            match self {
                Self::Pass(id) => json!(format!(r#"{{"id":{id}}}"#)),
                Self::ParseError => json!("not json"),
                Self::SchemaViolation(value) => json!(serde_json::to_string(&json!({"id": value}))
                    .expect("generated JSON must serialize")),
                Self::Skipped => Value::Null,
            }
        }

        fn expected_result(&self) -> ChoiceValidationResult {
            match self {
                Self::Pass(_) => ChoiceValidationResult::Pass,
                Self::ParseError => ChoiceValidationResult::JsonParseError {
                    byte_offset: 0,
                    expected: String::new(),
                },
                Self::SchemaViolation(_) => ChoiceValidationResult::SchemaViolations(Vec::new()),
                Self::Skipped => ChoiceValidationResult::Skipped,
            }
        }
    }

    fn generated_choice() -> impl Strategy<Value = GeneratedChoice> {
        prop_oneof![
            4 => any::<i64>().prop_map(GeneratedChoice::Pass),
            2 => Just(GeneratedChoice::ParseError),
            2 => "[a-zA-Z0-9 ]{1,24}".prop_map(GeneratedChoice::SchemaViolation),
            2 => Just(GeneratedChoice::Skipped),
        ]
    }

    fn assert_choice_result(actual: &ChoiceValidationResult, expected: &ChoiceValidationResult) {
        match expected {
            ChoiceValidationResult::Pass => assert_eq!(actual, &ChoiceValidationResult::Pass),
            ChoiceValidationResult::JsonParseError { .. } => assert!(matches!(
                actual,
                ChoiceValidationResult::JsonParseError { .. }
            )),
            ChoiceValidationResult::SchemaViolations(_) => assert!(matches!(
                actual,
                ChoiceValidationResult::SchemaViolations(violations) if !violations.is_empty()
            )),
            ChoiceValidationResult::Skipped => {
                assert_eq!(actual, &ChoiceValidationResult::Skipped)
            }
        }
    }

    fn expected_aggregate(choices: &[GeneratedChoice]) -> StructuredOutputOutcome {
        if choices.iter().any(|choice| {
            matches!(
                choice,
                GeneratedChoice::ParseError | GeneratedChoice::SchemaViolation(_)
            )
        }) {
            StructuredOutputOutcome::Fail
        } else if choices
            .iter()
            .any(|choice| matches!(choice, GeneratedChoice::Skipped))
        {
            StructuredOutputOutcome::Skipped
        } else {
            StructuredOutputOutcome::Pass
        }
    }

    fn passthrough_engine(
        provider: &str,
        model: &str,
        global_listed: bool,
        group_override: Option<bool>,
        provider_override: Option<bool>,
    ) -> StructuredOutputEngine {
        let global_provider = global_listed.then_some(provider);
        let group_providers = group_override.map(|listed| {
            if listed {
                vec![provider.to_owned()]
            } else {
                Vec::new()
            }
        });
        let mut config = config_with_structured_output(
            &format!(
                "  enabled: true\n  passthrough_providers: [{}]",
                global_provider.unwrap_or("")
            ),
            "  - name: group\n    models: []",
        );
        let group = &mut config.model_groups[0];
        group.structured_output = group_override.map(|_| StructuredOutputOverride {
            passthrough_providers: group_providers,
            ..StructuredOutputOverride::default()
        });
        group.models.push(ProviderModel {
            provider: provider.to_owned(),
            model: model.to_owned(),
            cost_per_million_input_tokens: 0.0,
            cost_per_million_output_tokens: 0.0,
            priority: 100,
            structured_output_passthrough: provider_override,
            tier: None,
            context_window: 0,
            specializations: vec![],
        });

        StructuredOutputEngine::from_config(&config).unwrap()
    }

    #[test]
    fn from_config_requires_top_level_section() {
        let config: Config = serde_yaml::from_str(
            r#"
server:
  host: 127.0.0.1
  port: 8080
providers: []
model_groups: []
"#,
        )
        .unwrap();

        assert!(StructuredOutputEngine::from_config(&config).is_none());
    }

    #[test]
    fn passthrough_precedence_is_provider_then_group_then_global() {
        let config = config_with_structured_output(
            "  enabled: true\n  max_retries: 1\n  passthrough_providers: [global, explicit-off]",
            r#"  - name: group
    structured_output:
      max_retries: 3
      retry_temperature: 0.7
      passthrough_providers: [group-provider, explicit-off]
    models:
      - provider: explicit-off
        model: model
        structured_output_passthrough: false
      - provider: explicit-on
        model: model
        structured_output_passthrough: true
      - provider: group-provider
        model: model
      - provider: global
        model: model"#,
        );
        let engine = StructuredOutputEngine::from_config(&config).unwrap();

        let group = engine.effective_config("group", "group-provider", "model");
        assert_eq!(group.max_retries, 3);
        assert_eq!(group.retry_temperature, 0.7);
        assert!(group.passthrough);
        assert!(
            !engine
                .effective_config("group", "global", "model")
                .passthrough
        );
        assert!(
            !engine
                .effective_config("group", "explicit-off", "model")
                .passthrough
        );
        assert!(
            engine
                .effective_config("group", "explicit-on", "model")
                .passthrough
        );
        assert!(
            engine
                .effective_config("unknown", "global", "model")
                .passthrough
        );
    }

    // Task 12.9: Passthrough precedence.
    // Explicit provider/model policy overrides group policy, group policy overrides
    // global policy, and absence at every level falls back to `false`.
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_passthrough_precedence_covers_all_policy_combinations(
            provider in "[a-z][a-z0-9-]{0,15}",
            model in "[a-z][a-z0-9-]{0,15}",
        ) {
            for global_listed in [false, true] {
                for group_override in [None, Some(false), Some(true)] {
                    for provider_override in [None, Some(false), Some(true)] {
                        let engine = passthrough_engine(
                            &provider,
                            &model,
                            global_listed,
                            group_override,
                            provider_override,
                        );
                        let expected = provider_override
                            .or(group_override)
                            .unwrap_or(global_listed);
                        let effective = engine.effective_config("group", &provider, &model);

                        prop_assert_eq!(effective.passthrough, expected);
                        prop_assert!(matches!(
                            engine.should_validate(&schema_request(), "group", &provider, &model),
                            ValidationDecision::Skipped(ValidationSkipReason::Passthrough)
                        ) == expected);
                        prop_assert_eq!(
                            engine
                                .effective_config("unknown", &provider, &model)
                                .passthrough,
                            global_listed
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn activation_decisions_distinguish_all_skip_paths() {
        let config = config_with_structured_output(
            "  enabled: true\n  passthrough_providers: [native]",
            r#"  - name: group
    models:
      - provider: active
        model: model
      - provider: native
        model: model
  - name: disabled
    structured_output:
      enabled: false
    models:
      - provider: active
        model: model"#,
        );
        let engine = StructuredOutputEngine::from_config(&config).unwrap();

        assert!(matches!(
            engine.should_validate(&request(None), "group", "active", "model"),
            ValidationDecision::NotApplicable
        ));
        assert!(matches!(
            engine.should_validate(&schema_request(), "disabled", "active", "model"),
            ValidationDecision::Skipped(ValidationSkipReason::Disabled)
        ));
        assert!(matches!(
            engine.should_validate(&schema_request(), "group", "native", "model"),
            ValidationDecision::Skipped(ValidationSkipReason::Passthrough)
        ));
        assert!(matches!(
            engine.should_validate(
                &request(Some(json!({"type": "json_schema"}))),
                "group",
                "active",
                "model"
            ),
            ValidationDecision::Skipped(ValidationSkipReason::Malformed(_))
        ));
        assert!(matches!(
            engine.should_validate(
                &request(Some(json!({
                    "type": "json_schema",
                    "json_schema": {"schema": {}}
                }))),
                "group",
                "active",
                "model"
            ),
            ValidationDecision::Skipped(ValidationSkipReason::CompileFailed(_))
        ));
        assert!(matches!(
            engine.should_validate(&schema_request(), "group", "active", "model"),
            ValidationDecision::Validate(_)
        ));

        let mut metrics = String::new();
        engine
            .metrics()
            .write_structured_output_prometheus(&mut metrics);
        assert!(metrics.contains("status=\"skip\"} 3"));
    }

    // Task 12.5: Multi-choice validation independence.
    // Each generated choice must retain its own validator result regardless of
    // neighboring choices, while the request outcome applies fail/skip/pass precedence.
    proptest! {
        #![proptest_config(ProptestConfig::with_cases(64))]

        #[test]
        fn prop_multi_choice_validation_is_independent(
            generated in proptest::collection::vec(generated_choice(), 1..=8),
        ) {
            let config = config_with_structured_output(
                "  enabled: true",
                "  - name: group\n    models:\n      - provider: active\n        model: model",
            );
            let engine = StructuredOutputEngine::from_config(&config).unwrap();
            let context = active_context(&engine);
            let expected_results: Vec<_> = generated
                .iter()
                .map(GeneratedChoice::expected_result)
                .collect();
            let expected_outcome = expected_aggregate(&generated);
            let generated_response = response(
                generated
                    .iter()
                    .map(GeneratedChoice::content)
                    .collect(),
            );
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_all()
                .build()
                .expect("test runtime must build");

            let actual = runtime.block_on(engine.validate_response(
                &context,
                &generated_response,
                "active",
                "model",
            ));

            prop_assert_eq!(actual.outcome, expected_outcome);
            prop_assert_eq!(actual.choices.len(), expected_results.len());
            for (choice, expected) in actual.choices.iter().zip(&expected_results) {
                assert_choice_result(&choice.result, expected);
                prop_assert!(choice.internal_skip.is_none());
            }
        }
    }

    #[tokio::test]
    async fn validation_records_one_counter_without_observing_request_latency() {
        let config = config_with_structured_output(
            "  enabled: true",
            "  - name: group\n    models:\n      - provider: active\n        model: model",
        );
        let engine = StructuredOutputEngine::from_config(&config).unwrap();
        let context = active_context(&engine);

        let result = engine
            .validate_response(
                &context,
                &response(vec![json!("{\"id\":1}")]),
                "active",
                "model",
            )
            .await;
        assert_eq!(result.outcome, StructuredOutputOutcome::Pass);

        let mut metrics = String::new();
        engine
            .metrics()
            .write_structured_output_prometheus(&mut metrics);
        assert!(metrics.contains("status=\"pass\"} 1"));
        assert!(!metrics.contains("obey_api_structured_output_latency_ms_count{"));
    }

    #[tokio::test]
    async fn multi_choice_validation_fails_then_skips_then_passes() {
        let config = config_with_structured_output(
            "  enabled: true",
            r#"  - name: group
    models:
      - provider: active
        model: model"#,
        );
        let engine = StructuredOutputEngine::from_config(&config).unwrap();
        let context = active_context(&engine);

        let failed = engine
            .validate_response(
                &context,
                &response(vec![json!("{\"id\":1}"), json!("not json"), Value::Null]),
                "active",
                "model",
            )
            .await;
        assert_eq!(failed.outcome, StructuredOutputOutcome::Fail);
        assert_eq!(failed.choices.len(), 3);
        assert_eq!(failed.choices[0].result, ChoiceValidationResult::Pass);
        assert!(matches!(
            failed.choices[1].result,
            ChoiceValidationResult::JsonParseError { .. }
        ));
        assert_eq!(failed.choices[2].result, ChoiceValidationResult::Skipped);

        let skipped = engine
            .validate_response(
                &context,
                &response(vec![json!("{\"id\":2}"), json!("   ")]),
                "active",
                "model",
            )
            .await;
        assert_eq!(skipped.outcome, StructuredOutputOutcome::Skipped);

        let passed = engine
            .validate_response(
                &context,
                &response(vec![json!("{\"id\":3}"), json!("{\"id\":4}")]),
                "active",
                "model",
            )
            .await;
        assert_eq!(passed.outcome, StructuredOutputOutcome::Pass);
    }

    #[test]
    fn cache_eligibility_allows_only_not_applicable_and_pass() {
        assert!(StructuredOutputEngine::is_cache_eligible(
            StructuredOutputOutcome::NotApplicable
        ));
        assert!(StructuredOutputEngine::is_cache_eligible(
            StructuredOutputOutcome::Pass
        ));
        assert!(!StructuredOutputEngine::is_cache_eligible(
            StructuredOutputOutcome::Fail
        ));
        assert!(!StructuredOutputEngine::is_cache_eligible(
            StructuredOutputOutcome::Skipped
        ));
    }
}
