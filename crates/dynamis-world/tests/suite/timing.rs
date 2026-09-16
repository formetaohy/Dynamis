use super::common::{DT, gravity_config, new_world, settle};
use dynamis_model::{BodyDesc, ConstraintDesc, SoftBodyDesc};

const QUERY_RUN: &[&str] = &["update_query_aabbs"];

const SCENE_INDEX: &[&str] = &["update_soft_bounds", "emit_soft_entries"];

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
    for label in [
        "prepare",
        "broadphase",
        "narrowphase",
        "solve_substeps",
        "commit",
    ] {
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
            (!timing.label.starts_with("soft_") || SCENE_INDEX.contains(&timing.label))
                && !timing.label.starts_with("ccd_"),
            "pass {} ran while its domain holds no work",
            timing.label
        );
    }
}

#[test]
fn a_soft_step_profiles_no_pass_of_an_absent_domain() {
    let mut world = new_world(gravity_config());
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
    settle(&mut world, 30);
    let timings = world.gpu_pass_timings();
    assert!(
        timings
            .iter()
            .any(|timing| timing.label == "solve_soft_substeps"),
        "a free soft body must simulate"
    );
    for timing in timings {
        assert!(
            !matches!(
                timing.label,
                "narrowphase"
                    | "build_islands"
                    | "wake"
                    | "live"
                    | "solver_prepare"
                    | "solve_substeps"
                    | "ccd_sweep"
                    | "ccd_apply"
                    | "apply_reactions"
                    | "sleep"
                    | "resting_gather"
                    | "resting_index"
            ),
            "pass {} ran while the world holds no rigid body",
            timing.label
        );
    }
}

#[test]
fn a_full_scene_profiles_every_pass_of_a_step() {
    let mut world = new_world(gravity_config());
    world.spawn(BodyDesc::sphere(0.4).position([0.0, 4.0, 0.0]).ccd(true));
    let anchor = world.spawn(BodyDesc::static_sphere(0.2).position([0.0, 3.0, 0.0]));
    let arm = world.spawn(BodyDesc::sphere(0.2).position([1.0, 3.0, 0.0]));
    let joint = world.add_constraint(
        anchor,
        arm,
        ConstraintDesc::revolute([0.0; 3], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    );
    world.try_joint_state(joint);
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
    world.set_gravity([0.0, -10.0, 0.0]);
    world.step(DT);
    world.wait();
    let mut declared: Vec<&str> = world
        .pass_labels()
        .into_iter()
        .filter(|label| !QUERY_RUN.contains(label))
        .collect();
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
        "every pass the schedule declares for a step must be profiled while the scene speaks to every domain"
    );
}

#[test]
fn a_profiled_world_keeps_reporting_after_a_snapshot() {
    let mut world = new_world(gravity_config());
    for index in 0..8 {
        world.spawn(BodyDesc::sphere(0.4).position([index as f32, 2.0, 0.0]));
    }
    settle(&mut world, 12);
    let snapshot = world.snapshot();
    settle(&mut world, 12);
    world.restore(&snapshot);
    settle(&mut world, 12);
    let timings = world.gpu_pass_timings();
    assert!(
        timings
            .iter()
            .any(|timing| timing.label == "solve_substeps"),
        "a restored world must keep profiling the passes it steps"
    );
}

#[test]
fn a_query_on_a_sleeping_world_profiles_only_the_index() {
    let mut world = new_world(gravity_config());
    world.spawn(
        BodyDesc::cuboid([5.0, 0.5, 5.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    for index in 0..16 {
        world.spawn(BodyDesc::sphere(0.4).position([
            (index % 4) as f32 * 0.85 - 1.3,
            0.4,
            (index / 4) as f32 * 0.85 - 1.3,
        ]));
    }
    settle(&mut world, 400);

    for _ in 0..4 {
        let _ = world.ray_query(
            [0.0, 4.0, 0.0],
            [0.0, -1.0, 0.0],
            20.0,
            &dynamis_model::QueryFilter::default(),
        );
        world.step(super::common::DT);
    }
    world.wait();

    let timings = world.gpu_pass_timings();
    for timing in timings {
        assert!(
            !matches!(
                timing.label,
                "narrowphase"
                    | "build_islands"
                    | "wake"
                    | "live"
                    | "solver_prepare"
                    | "solve_substeps"
                    | "sleep"
                    | "resting_gather"
                    | "resting_index"
            ),
            "pass {} ran while the world only had a query pending",
            timing.label
        );
    }
    assert!(
        timings.iter().any(|timing| timing.label == "broadphase"),
        "a query must still refresh the broadphase index"
    );
    assert!(
        timings.iter().any(|timing| timing.label == "query"),
        "a query must resolve inside the declared query pass"
    );
}
