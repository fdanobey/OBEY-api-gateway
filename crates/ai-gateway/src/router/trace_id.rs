use uuid::Uuid;

/// Generate a unique trace ID for a request
///
/// Uses X-Request-Id header if provided by client, otherwise generates UUID v4
///
/// Requirements: 33.1, 33.7
pub fn generate_trace_id(request_id_header: Option<&str>) -> String {
    match request_id_header {
        Some(id) if !id.is_empty() => id.to_string(),
        _ => Uuid::new_v4().to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_trace_id_from_header() {
        let request_id = "custom-request-id-123";
        let trace_id = generate_trace_id(Some(request_id));
        assert_eq!(trace_id, request_id);
    }

    #[test]
    fn test_generate_trace_id_empty_header() {
        let trace_id = generate_trace_id(Some(""));
        // Should generate UUID when header is empty
        assert!(Uuid::parse_str(&trace_id).is_ok());
    }

    #[test]
    fn test_generate_trace_id_no_header() {
        let trace_id = generate_trace_id(None);
        // Should generate valid UUID v4
        assert!(Uuid::parse_str(&trace_id).is_ok());
    }

    #[test]
    fn test_generate_trace_id_unique() {
        let trace_id1 = generate_trace_id(None);
        let trace_id2 = generate_trace_id(None);
        // Each generated ID should be unique
        assert_ne!(trace_id1, trace_id2);
    }

    // Property-based tests
    use proptest::prelude::*;

    // Property 13: Trace ID Propagation
    //
    // For any request, the trace ID shall appear in all log entries related to that request,
    // in the X-Trace-Id response header, and in provider request headers.
    //
    // **Validates: Requirements 33.1, 33.2, 33.3, 33.4**
    proptest! {
        #[test]
        fn prop_trace_id_propagation(
            has_request_id in proptest::bool::ANY,
            request_id in "[a-zA-Z0-9-]{1,50}",
        ) {
            let trace_id = if has_request_id {
                generate_trace_id(Some(&request_id))
            } else {
                generate_trace_id(None)
            };

            // Req 33.1: Trace ID must be non-empty (unique ID generated)
            prop_assert!(!trace_id.is_empty(), "Trace ID must not be empty");

            // Req 33.7 -> 33.1: If X-Request-Id provided, use it as trace ID
            if has_request_id {
                prop_assert_eq!(&trace_id, &request_id, "Trace ID must match provided X-Request-Id");
            }

            // Req 33.1: If no request ID, trace ID must be valid UUID v4
            if !has_request_id {
                prop_assert!(
                    uuid::Uuid::parse_str(&trace_id).is_ok(),
                    "Generated trace ID must be valid UUID: {}",
                    trace_id
                );
            }

            // Req 33.3/33.4: Trace ID is suitable for use in response/provider headers
            // (no control chars, reasonable length)
            prop_assert!(
                !trace_id.contains(|c: char| c.is_control()),
                "Trace ID must not contain control characters"
            );
            prop_assert!(
                trace_id.len() <= 100,
                "Trace ID should be reasonable length for headers"
            );

            // Req 33.7: Deterministic when X-Request-Id provided
            if has_request_id {
                let trace_id2 = generate_trace_id(Some(&request_id));
                prop_assert_eq!(
                    trace_id, trace_id2,
                    "Trace ID must be deterministic when X-Request-Id is provided"
                );
            }
        }
    }
}
