use ai_gateway::loop_detection::metrics::LoopDetectionMetrics;

#[test]
fn exact_loop_metric_names_are_exposed() {
    let metrics = LoopDetectionMetrics::new();
    metrics.record_confidence(Some("vk-id"), 0.75);
    metrics.record_enforcement("warn", Some("vk-id"));
    metrics.record_eviction();
    let mut output = String::new();
    metrics.write_prometheus(&mut output, 3);

    assert!(output.contains("# TYPE obey_loop_confidence_score histogram"));
    assert!(output.contains("obey_loop_confidence_score_bucket{virtual_key=\"id:vk-id\",le=\"0.8\"} 1"));
    assert!(output.contains("# TYPE obey_loop_enforcement_total counter"));
    assert!(output.contains("obey_loop_enforcement_total{level=\"warn\",virtual_key=\"id:vk-id\"} 1"));
    assert!(output.contains("obey_loop_sessions_active 3"));
    assert!(output.contains("obey_loop_sessions_evicted_total 1"));
    assert_eq!(metrics.evicted_total(), 1);
    assert!((metrics.evictions_per_minute() - 0.2).abs() < f64::EPSILON);
    assert!(!output.contains("Bearer"));
}

#[test]
fn metric_initialization_is_infallible_and_empty_histogram_is_valid() {
    let metrics = LoopDetectionMetrics::new();
    let mut output = String::new();
    metrics.write_prometheus(&mut output, 0);
    assert!(output.contains("obey_loop_sessions_active 0"));
    assert!(output.contains("obey_loop_sessions_evicted_total 0"));
}
