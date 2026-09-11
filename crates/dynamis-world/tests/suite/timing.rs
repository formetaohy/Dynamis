use super::common::{gravity_config, new_world, settle};
use dynamis_model::BodyDesc;

#[test]
fn a_stepped_world_reports_one_duration_per_pass() {
    let mut world = new_world(gravity_config());
    for index in 0..32 {
        world.spawn(BodyDesc::sphere(0.4).mass(1.0).position([
            (index % 8) as f32 * 0.9,
            2.0 + (index / 8) as f32,
            0.0,
        ]));
    }
    assert!(
        world.gpu_timing_supported(),
        "the engine requests timestamp queries, so a profile build must report them"
    );
    settle(&mut world, 30);
    let timings = world.gpu_pass_timings();
    assert!(
        !timings.is_empty(),
        "a drained step must leave per-pass timings behind"
    );
    for label in ["integrate", "broadphase", "narrowphase", "commit"] {
        assert!(
            timings.iter().any(|timing| timing.label == label),
            "pass {label} must be attributed"
        );
    }
    let mut labels: Vec<&str> = timings.iter().map(|timing| timing.label).collect();
    labels.sort_unstable();
    let unique = labels.len();
    labels.dedup();
    assert_eq!(labels.len(), unique, "pass labels must be reported once");
    for timing in timings {
        assert!(
            timing.nanoseconds >= 0.0,
            "{} measured a negative duration",
            timing.label
        );
    }
}
