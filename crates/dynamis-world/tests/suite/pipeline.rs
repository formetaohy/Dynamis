use super::common::{
    DT, gravity_config, observed_world, settle, settle_until, static_sphere_ground,
};
use dynamis_model::BodyDesc;

const STEP: &[&str] = &[
    "apply_commands",
    "wake_all",
    "character",
    "prepare",
    "update_query_aabbs",
    "vehicle",
    "update_soft_bounds",
    "emit_entries",
    "emit_soft_entries",
    "broadphase",
    "narrowphase",
    "build_islands",
    "wake",
    "live",
    "solver_prepare",
    "solve_substeps",
    "ccd_sweep",
    "ccd_apply",
    "apply_soft_inputs",
    "soft_wake_all",
    "soft_settle",
    "solve_soft_substeps",
    "emit_contact_facts",
    "apply_reactions",
    "sleep",
    "commit",
    "consume_streams",
    "observe",
    "observe_joints",
    "resting_gather",
    "resting_index",
    "query",
    "sweep_characters",
    "sweep_vehicles",
];

const IDLE: &[&str] = &[
    "apply_commands",
    "commit",
    "consume_streams",
    "observe",
    "query",
];

#[test]
fn the_step_resolves_the_declared_domain_coupling() {
    let world = observed_world(gravity_config());
    assert_eq!(world.pass_labels(), STEP);
}

#[test]
fn an_awake_world_runs_the_index_and_the_simulation() {
    let mut world = observed_world(gravity_config());
    world.spawn(BodyDesc::sphere(0.5).position([0.0, 5.0, 0.0]));
    settle(&mut world, 2);
    let ran = world.ran_passes();
    for pass in [
        "emit_entries",
        "broadphase",
        "narrowphase",
        "solve_substeps",
        "sleep",
    ] {
        assert!(
            ran.contains(&pass),
            "an awake step must run {pass}, ran {ran:?}"
        );
    }
}

#[test]
fn an_idle_world_runs_only_the_unconditional_passes() {
    let mut world = observed_world(gravity_config());
    let _floor = static_sphere_ground(&mut world, 1.0);
    let ball = world.spawn(BodyDesc::sphere(0.5).position([0.0, 1.5, 0.0]));
    settle_until(&mut world, 600, |world| {
        world.read_state(ball).sleeping && world.ran_passes() == IDLE
    });
    assert_eq!(world.ran_passes(), IDLE);
}

#[test]
fn a_burst_outgrows_the_measured_work_it_streams_over() {
    let mut world = observed_world(gravity_config());
    let _ground = world.spawn(
        BodyDesc::cuboid([40.0, 0.5, 40.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let lone = world.spawn(BodyDesc::sphere(0.2).position([-38.0, 3.0, -38.0]));
    settle(&mut world, 4);
    assert!(world.read_state(lone).position[1] < 3.0);
    let mut burst = Vec::new();
    for index in 0..2048 {
        let x = (index % 64) as f32 - 32.0;
        let z = (index / 64) as f32 - 16.0;
        burst.push(world.spawn(BodyDesc::sphere(0.2).position([x, 3.0, z])));
    }
    world.step(DT);
    world.wait();
    let stalled = burst
        .iter()
        .filter(|handle| world.read_state(**handle).position[1] >= 3.0)
        .count();
    assert_eq!(
        stalled, 0,
        "a burst must integrate past its quiet measurement"
    );
    settle(&mut world, 60);
    let sunk = burst
        .iter()
        .filter(|handle| world.read_state(**handle).position[1] <= 0.05)
        .count();
    assert_eq!(
        sunk, 0,
        "a burst must land on a ground it was never measured with"
    );
}
