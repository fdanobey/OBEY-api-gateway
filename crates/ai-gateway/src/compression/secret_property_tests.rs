use super::engines::rtk::RtkEngine;
use super::{CompressiblePayload, CompressionContext, CompressionEngine};
use crate::models::openai::{Message, OpenAIRequest};
use proptest::prelude::*;
use proptest::test_runner::TestCaseResult;
use serde_json::{Map, Value};

const REDACTED: &str = "[REDACTED]";

fn normal_fragment() -> impl Strategy<Value = String> {
    proptest::string::string_regex("[a-z]{3,12}(?: [a-z]{3,12}){1,4}")
        .expect("normal fragment regex must compile")
}

fn openai_secret() -> impl Strategy<Value = String> {
    proptest::string::string_regex("[A-Za-z0-9_-]{18,36}")
        .expect("OpenAI secret regex must compile")
        .prop_map(|body| format!("sk-Aa{body}9"))
}

fn aws_secret() -> impl Strategy<Value = String> {
    proptest::string::string_regex("[A-Z0-9]{16}")
        .expect("AWS secret regex must compile")
        .prop_map(|body| format!("AKIA{body}"))
}

fn bearer_secret() -> impl Strategy<Value = String> {
    proptest::string::string_regex("[A-Za-z0-9._~+/=-]{12,36}")
        .expect("bearer secret regex must compile")
        .prop_map(|body| format!("aA._~+/-={body}Z9"))
}

fn bearer_scheme() -> impl Strategy<Value = &'static str> {
    prop_oneof![Just("Bearer"), Just("bEaReR"), Just("BEARER")]
}

fn url_username() -> impl Strategy<Value = String> {
    proptest::string::string_regex("[a-z][a-z0-9]{3,12}").expect("URL username regex must compile")
}

fn url_password() -> impl Strategy<Value = String> {
    proptest::string::string_regex("[A-Za-z0-9_-]{12,28}")
        .expect("URL password regex must compile")
        .prop_map(|body| format!("Pp-{body}-9"))
}

fn command_output_request(role: &str, output: String) -> OpenAIRequest {
    let mut message_extra = Map::new();
    message_extra.insert(
        "tool_call_id".to_owned(),
        Value::String("property-secret-command".to_owned()),
    );
    message_extra.insert(
        "command".to_owned(),
        Value::String("cargo build --release".to_owned()),
    );
    message_extra.insert("name".to_owned(), Value::String("terminal".to_owned()));

    OpenAIRequest {
        model: "gpt-4o".to_owned(),
        messages: vec![Message {
            role: role.to_owned(),
            content: Value::String(output),
            extra: message_extra,
        }],
        stream: false,
        temperature: None,
        max_tokens: None,
        extra: Map::new(),
    }
}

#[allow(clippy::too_many_arguments)]
fn assert_secret_redaction(
    role: &str,
    fragments: Vec<String>,
    openai: String,
    aws: String,
    bearer_scheme: &str,
    bearer: String,
    username: String,
    password: String,
) -> TestCaseResult {
    let credential_url = format!("https://{username}:{password}@example.test/artifact");
    let input = [
        format!("build started: {}", fragments[0]),
        format!(
            "warning: {} openai credential {openai} {}",
            fragments[1], fragments[2]
        ),
        format!(
            "error: {} aws credential {aws} {}",
            fragments[3], fragments[4]
        ),
        format!(
            "warning: {} authorization {bearer_scheme} {bearer} {}",
            fragments[5], fragments[6]
        ),
        format!(
            "error: {} registry {credential_url} {}",
            fragments[7], fragments[8]
        ),
        format!("build failed: {}", fragments[9]),
    ]
    .join("\n");
    let expected = [
        format!("build started: {}", fragments[0]),
        format!(
            "warning: {} openai credential {REDACTED} {}",
            fragments[1], fragments[2]
        ),
        format!(
            "error: {} aws credential {REDACTED} {}",
            fragments[3], fragments[4]
        ),
        format!(
            "warning: {} authorization Bearer {REDACTED} {}",
            fragments[5], fragments[6]
        ),
        format!(
            "error: {} registry https://{username}:{REDACTED}@example.test/artifact {}",
            fragments[7], fragments[8]
        ),
        format!("build failed: {}", fragments[9]),
    ]
    .join("\n");

    let context = CompressionContext::new("gpt-4o", "property-test");
    let request = command_output_request(role, input.clone());
    let input_tokens = context.token_counter.count_request(&request);
    let mut payload = CompressiblePayload::from_openai_request(request);
    prop_assert_eq!(payload.messages[0].role.as_str(), role);
    prop_assert!(!payload.messages[0].cache_protected);
    prop_assert!(payload.messages[0].content.as_text().is_some());
    prop_assert!(!payload.messages[0]
        .relationships
        .tool_result_for_ids
        .is_empty());

    let runtime = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("property-test runtime must build");
    let result = runtime.block_on(RtkEngine::new().compress(&mut payload, &context));
    let output = payload.messages[0]
        .content
        .as_text()
        .expect("textual command output must remain textual");
    let output_tokens = context
        .token_counter
        .count_request(&payload.clone().into_openai_request());

    prop_assert!(result.applied);
    prop_assert_eq!(result.tokens_before, input_tokens);
    prop_assert_eq!(result.tokens_after, output_tokens);
    prop_assert!(output_tokens <= input_tokens);
    prop_assert_eq!(output, expected);
    prop_assert_eq!(output.matches(REDACTED).count(), 4);
    prop_assert!(!output.contains(&openai));
    prop_assert!(!output.contains(&aws));
    prop_assert!(!output.contains(&bearer));
    prop_assert!(!output.contains(&password));
    prop_assert!(!output.contains(&credential_url));
    for fragment in fragments {
        prop_assert!(output.contains(&fragment));
    }

    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(64))]

    #[test]
    fn property_9_command_output_secret_redaction(
        role in prop_oneof![Just("tool"), Just("assistant")],
        fragments in prop::collection::vec(normal_fragment(), 10),
        openai in openai_secret(),
        aws in aws_secret(),
        bearer_scheme in bearer_scheme(),
        bearer in bearer_secret(),
        username in url_username(),
        password in url_password(),
    ) {
        assert_secret_redaction(
            role,
            fragments,
            openai,
            aws,
            bearer_scheme,
            bearer,
            username,
            password,
        )?;
    }
}
