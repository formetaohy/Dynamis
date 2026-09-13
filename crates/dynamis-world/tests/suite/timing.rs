use super::common::{gravity_config, new_world, settle};
use dynamis_model::{BodyDesc, SoftBodyDesc};

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
        "the pass schedule requests timestamp queries, so a profile build must report them"
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

#[test]
fn a_rigid_step_profiles_no_pass_of_an_absent_domain() {
    let mut world = new_world(gravity_config());
    for index in 0..8 {
        world.spawn(BodyDesc::sphere(0.4).position([index as f32, 2.0, 0.0]));
    }
    settle(&mut world, 30);
    let timings = world.gpu_pass_timings();
    assert!(
        !timings.is_empty(),
        "a drained step must leave per-pass timings behind"
    );
    for timing in timings {
        assert!(
            !timing.label.starts_with("soft_") && !timing.label.starts_with("ccd_"),
            "pass {} ran while its domain holds no work",
            timing.label
        );
    }
}

#[test]
fn a_full_scene_profiles_every_declared_pass() {
    let mut world = new_world(gravity_config());
    world.spawn(BodyDesc::sphere(0.4).position([0.0, 4.0, 0.0]).ccd(true));
    world.add_soft_body(
        SoftBodyDesc::net(
            vec![
                [-0.25, 0.0, -0.25],
                [0.25, 0.0, -0.25],
                [0.25, 0.0, 0.25],
                [-0.25, 0.0, 0.25],
            ],
            vec![[0, 1], [1, 2], [2, 3], [3, 0]],
        )
        .radius(0.1)
        .position([0.0, 2.0, 0.0]),
    );
    settle(&mut world, 12);
    let mut declared = world.pass_labels();
    declared.sort_unstable();
    let mut reported: Vec<&str> = world
        .gpu_pass_timings()
        .iter()
        .map(|timing| timing.label)
        .collect();
    reported.sort_unstable();
    reported.dedup();
    assert_eq!(
        reported, declared,
        "every pass the schedule declares must be profiled while the scene speaks to every domain"
    );
}
