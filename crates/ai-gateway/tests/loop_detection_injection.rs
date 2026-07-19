use ai_gateway::{
    loop_detection::{
        enforcement::{EnforcementDecision, EnforcementEngine},
        injection::{InjectionEngine, DEFAULT_BREAK_INSTRUCTION, ERROR_CYCLING_INSTRUCTION},
        EnforcementLevel, InjectionStrategy, LoopDetectionConfig, SessionState,
    },
    models::openai::{Message, OpenAIRequest},
};
use serde_json::{json, Map, Value};

fn request(messages: Vec<Message>) -> OpenAIRequest {
    OpenAIRequest {
        model: "gpt-4".into(),
        messages,
        stream: false,
        temperature: None,
        max_tokens: None,
        extra: Map::new(),
    }
}

fn decision(signal: &'static str) -> EnforcementDecision {
    EnforcementDecision {
        level: EnforcementLevel::Inject,
        transitioned: true,
        dominant_signal: signal,
        should_warn: true,
        should_throttle: true,
        should_inject: true,
        should_hard_stop: false,
    }
}

#[test]
fn appends_to_existing_system_prompt_once() {
    let mut request = request(vec![Message {
        role: "system".into(),
        content: json!("Existing policy"),
        extra: Map::new(),
    }]);
    let mut session = SessionState::new(None, 5);
    assert!(InjectionEngine::inject(
        &mut request,
        &decision("content_similarity"),
        &mut session,
        &LoopDetectionConfig::default(),
        None,
    ));
    assert_eq!(
        request.messages[0].content.as_str().unwrap(),
        format!("Existing policy\n\n{DEFAULT_BREAK_INSTRUCTION}")
    );
    assert!(!InjectionEngine::inject(
        &mut request,
        &decision("content_similarity"),
        &mut session,
        &LoopDetectionConfig::default(),
        None,
    ));
}

#[test]
fn creates_system_message_when_absent() {
    let mut request = request(vec![Message {
        role: "user".into(),
        content: json!("hello"),
        extra: Map::new(),
    }]);
    let mut session = SessionState::new(None, 5);
    InjectionEngine::inject(
        &mut request,
        &decision("content_similarity"),
        &mut session,
        &LoopDetectionConfig::default(),
        None,
    );
    assert_eq!(request.messages[0].role, "system");
    assert_eq!(request.messages[0].content, json!(DEFAULT_BREAK_INSTRUCTION));
}

#[test]
fn context_aware_variants_and_custom_template_are_selected() {
    let mut tool_extra = Map::new();
    tool_extra.insert(
        "tool_calls".into(),
        json!([{"function":{"name":"read_file","arguments":"{}"}}]),
    );
    let mut config = LoopDetectionConfig {
        injection_strategy: InjectionStrategy::ContextAware,
        ..Default::default()
    };
    let mut tool_request = request(vec![Message {
        role: "assistant".into(),
        content: Value::Null,
        extra: tool_extra,
    }]);
    let mut session = SessionState::new(None, 5);
    InjectionEngine::inject(
        &mut tool_request,
        &decision("tool_call_repetition"),
        &mut session,
        &config,
        None,
    );
    assert!(tool_request.messages[0]
        .content
        .as_str()
        .unwrap()
        .contains("Stop calling read_file"));

    let mut error_request = request(vec![]);
    let mut session = SessionState::new(None, 5);
    InjectionEngine::inject(
        &mut error_request,
        &decision("error_cycling"),
        &mut session,
        &config,
        None,
    );
    assert_eq!(error_request.messages[0].content, json!(ERROR_CYCLING_INSTRUCTION));

    config.break_instruction_template = Some("custom escape".into());
    let mut custom_request = request(vec![]);
    let mut session = SessionState::new(None, 5);
    InjectionEngine::inject(
        &mut custom_request,
        &decision("error_cycling"),
        &mut session,
        &config,
        None,
    );
    assert_eq!(custom_request.messages[0].content, json!("custom escape"));
}

#[test]
fn enforcement_marks_injection_pending_until_engine_injects() {
    let mut config = LoopDetectionConfig::default();
    config.consecutive_counts.warn = 1;
    config.consecutive_counts.throttle = 1;
    config.consecutive_counts.inject = 1;
    let mut session = SessionState::new(None, 5);
    EnforcementEngine::evaluate(1.0, &mut session, &config);
    EnforcementEngine::evaluate(1.0, &mut session, &config);
    let transition = EnforcementEngine::evaluate(1.0, &mut session, &config);
    assert!(transition.should_inject);
    assert!(!session.injected_at_level);
}
