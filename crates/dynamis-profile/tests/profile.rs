use dynamis_profile::{Profiler, summary_of};
use std::time::Duration;

#[test]
fn summary_aggregates_statistics() {
    let summary = summary_of(&[100.0, 200.0, 300.0, 400.0, 500.0]);
    assert_eq!(summary.mean, 300.0);
    assert_eq!(summary.min, 100.0);
    assert_eq!(summary.max, 500.0);
    assert_eq!(summary.p95, 500.0);
}

#[test]
fn summary_handles_single_sample() {
    let summary = summary_of(&[42.0]);
    assert_eq!(summary.mean, 42.0);
    assert_eq!(summary.min, 42.0);
    assert_eq!(summary.max, 42.0);
    assert_eq!(summary.p95, 42.0);
}

#[test]
fn profiler_groups_samples_by_phase() {
    let mut profiler = Profiler::new();
    profiler.measure("work", || 1 + 1);
    profiler.measure("work", || 2 + 2);
    profiler.measure("other", || 0);
    let work = profiler.phase("work").expect("work phase exists");
    assert_eq!(work.count(), 2);
    assert_eq!(
        profiler.phase("other").expect("other phase exists").count(),
        1
    );
    assert_eq!(profiler.phases().len(), 2);
}

#[test]
fn measure_with_returns_value_and_duration() {
    let mut profiler = Profiler::new();
    let (value, elapsed) = profiler.measure_with("op", || 7);
    assert_eq!(value, 7);
    assert!(elapsed >= Duration::ZERO);
    let op = profiler.phase("op").expect("op phase exists");
    assert_eq!(op.count(), 1);
}
