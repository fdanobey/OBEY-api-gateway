//! Mock provider servers for integration testing.
//!
//! Provides wiremock-based mock servers that simulate OpenAI-compatible
//! provider behavior including success, errors, rate limits, and timeouts.

#![allow(dead_code)]

use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, ResponseTemplate};

/// Behavior to configure on a mock provider server.
#[derive(Debug, Clone)]
pub enum MockBehavior {
    /// Return a valid chat completion response.
    Success,
    /// Return the specified HTTP error status code.
    Error(u16),
    /// Return 429 Too Many Requests (rate limit).
    RateLimit,
    /// Delay the response beyond a reasonable timeout (simulates timeout).
    Timeout,
}

/// Returns a valid OpenAI chat completion JSON response body.
pub fn mock_chat_completion_response() -> serde_json::Value {
    serde_json::json!({
        "id": "chatcmpl-mock-12345",
        "object": "chat.completion",
        "created": 1700000000_i64,
        "model": "gpt-4",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello from mock provider!"
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 8,
            "total_tokens": 18
        }
    })
}

/// Returns an OpenAI-style error response body.
pub fn mock_error_response(status: u16, message: &str) -> serde_json::Value {
    serde_json::json!({
        "error": {
            "message": message,
            "type": "server_error",
            "code": status
        }
    })
}

/// Start a wiremock `MockServer` configured with the given behavior for
/// `POST /v1/chat/completions`.
///
/// Returns the running `MockServer` — call `.uri()` to get the base URL.
pub async fn start_mock_openai(behavior: MockBehavior) -> MockServer {
    let server = MockServer::start().await;

    let response = match behavior {
        MockBehavior::Success => {
            ResponseTemplate::new(200).set_body_json(mock_chat_completion_response())
        }

        MockBehavior::Error(status) => ResponseTemplate::new(status).set_body_json(
            mock_error_response(status, &format!("Mock error with status {}", status)),
        ),

        MockBehavior::RateLimit => ResponseTemplate::new(429)
            .set_body_json(mock_error_response(429, "Rate limit exceeded"))
            .append_header("retry-after", "1"),

        MockBehavior::Timeout => {
            // Delay long enough that any reasonable request timeout fires first.
            // The test client timeout is 200ms, so 500ms is more than sufficient.
            ResponseTemplate::new(200)
                .set_body_json(mock_chat_completion_response())
                .set_delay(std::time::Duration::from_millis(500))
        }
    };

    Mock::given(method("POST"))
        .and(path("/v1/chat/completions"))
        .respond_with(response)
        .mount(&server)
        .await;

    server
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_success_returns_200() {
        let server = start_mock_openai(MockBehavior::Success).await;

        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{}/v1/chat/completions", server.uri()))
            .json(&serde_json::json!({
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .send()
            .await
            .unwrap();

        assert_eq!(resp.status(), 200);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert_eq!(
            body["choices"][0]["message"]["content"],
            "Hello from mock provider!"
        );
    }

    #[tokio::test]
    async fn test_mock_error_returns_status() {
        let server = start_mock_openai(MockBehavior::Error(503)).await;

        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{}/v1/chat/completions", server.uri()))
            .json(&serde_json::json!({
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .send()
            .await
            .unwrap();

        assert_eq!(resp.status(), 503);
        let body: serde_json::Value = resp.json().await.unwrap();
        assert!(body["error"]["message"].as_str().unwrap().contains("503"));
    }

    #[tokio::test]
    async fn test_mock_rate_limit_returns_429() {
        let server = start_mock_openai(MockBehavior::RateLimit).await;

        let client = reqwest::Client::new();
        let resp = client
            .post(format!("{}/v1/chat/completions", server.uri()))
            .json(&serde_json::json!({
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .send()
            .await
            .unwrap();

        assert_eq!(resp.status(), 429);
        assert!(resp.headers().get("retry-after").is_some());
    }

    #[tokio::test]
    async fn test_mock_timeout_exceeds_short_deadline() {
        let server = start_mock_openai(MockBehavior::Timeout).await;

        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_millis(200))
            .build()
            .unwrap();

        let result = client
            .post(format!("{}/v1/chat/completions", server.uri()))
            .json(&serde_json::json!({
                "model": "gpt-4",
                "messages": [{"role": "user", "content": "hi"}]
            }))
            .send()
            .await;

        assert!(result.is_err(), "Expected timeout error");
    }
}
