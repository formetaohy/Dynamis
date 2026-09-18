use super::common::{DT, observed_world, settle};
use dynamis_model::{BodyDesc, PhysicsConfig};

const FRAMES: u32 = 60;

#[test]
fn a_driven_body_keeps_the_velocity_it_declares() {
    let mut world = observed_world(PhysicsConfig::default());
    let platform = world.spawn(
        BodyDesc::cuboid([1.0, 0.1, 1.0])
            .kinematic(true)
            .velocity([1.5, -0.25, 0.75])
            .angular_velocity([0.0, 2.0, 0.0]),
    );
    for _ in 0..FRAMES {
        world.step(DT);
    }
    world.wait();
    let state = world.read_state(platform);
    assert_eq!(
        state.velocity,
        [1.5, -0.25, 0.75],
        "a driven body declares the velocity it moves at"
    );
    assert_eq!(state.angular_velocity, [0.0, 2.0, 0.0]);
    let travelled = 1.5 * DT * FRAMES as f32;
    assert!(
        (state.position[0] - travelled).abs() < 1e-4,
        "a driven body must advance exactly by its declaration, got {}",
        state.position[0]
    );
}

#[test]
fn a_driven_body_declares_motion_beyond_the_velocity_limits() {
    let mut world = observed_world(PhysicsConfig {
        damping: 0.5,
        angular_damping: 0.5,
        max_velocity: 1.0,
        max_angular_velocity: 1.0,
        ..PhysicsConfig::default()
    });
    let platform = world.spawn(
        BodyDesc::cuboid([0.5; 3])
            .kinematic(true)
            .velocity([20.0, 0.0, 0.0])
            .angular_velocity([0.0, 5.0, 0.0]),
    );
    for _ in 0..10 {
        world.step(DT);
    }
    world.wait();
    let state = world.read_state(platform);
    assert_eq!(
        state.velocity,
        [20.0, 0.0, 0.0],
        "only a simulated body is limited to the configured velocity"
    );
    assert_eq!(state.angular_velocity, [0.0, 5.0, 0.0]);
}

#[test]
fn a_driven_body_keeps_its_declaration_while_it_carries_a_load() {
    let mut world = observed_world(PhysicsConfig::default());
    world.spawn(
        BodyDesc::cuboid([20.0, 0.5, 20.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let platform = world.spawn(
        BodyDesc::cuboid([1.5, 0.1, 1.5])
            .kinematic(true)
            .position([0.0, 0.1, 0.0])
            .velocity([1.0, 0.0, 0.0]),
    );
    let rider = world.spawn(BodyDesc::cuboid([0.4; 3]).position([0.0, 0.4, 0.0]));
    settle(&mut world, 90);
    assert_eq!(
        world.read_state(platform).velocity,
        [1.0, 0.0, 0.0],
        "a carried load must not rewrite the motion its carrier declares"
    );
    assert!(
        world.read_state(rider).position[0] > 0.5,
        "the declared motion must carry the load it touches, got {:?}",
        world.read_state(rider).position
    );
}

#[test]
fn a_driven_body_is_never_slept() {
    let mut world = observed_world(PhysicsConfig::default());
    let platform = world.spawn(
        BodyDesc::cuboid([1.0, 0.1, 1.0])
            .kinematic(true)
            .velocity([0.0, 0.0, 1.0]),
    );
    settle(&mut world, 180);
    let state = world.read_state(platform);
    assert!(
        !state.sleeping,
        "a driven body declares motion, so it never rests"
    );
    assert_eq!(state.velocity, [0.0, 0.0, 1.0]);
}
