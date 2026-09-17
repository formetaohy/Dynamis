use super::common::{DT, gravity_config, observed_world};
use dynamis_model::{BodyDesc, BodyHandle, ConstraintDesc, PhysicsConfig};
use dynamis_world::World;

fn speed_of(velocity: [f32; 3]) -> f32 {
    (velocity[0] * velocity[0] + velocity[1] * velocity[1] + velocity[2] * velocity[2]).sqrt()
}

fn swing_chain(world: &mut World, links: usize) -> Vec<BodyHandle> {
    let anchor = world.spawn(BodyDesc::static_sphere(0.05).position([0.0, 30.0, 0.0]));
    let mut handles = Vec::new();
    let mut previous = anchor;
    for index in 0..links {
        let link = world.spawn(
            BodyDesc::sphere(0.05)
                .position([1.0 + index as f32, 30.0, 0.0])
                .mass(1.0),
        );
        world.add_constraint(
            previous,
            link,
            ConstraintDesc::distance([0.0; 3], [0.0; 3], 1.0),
        );
        handles.push(link);
        previous = link;
    }
    handles
}

#[test]
fn a_rotating_joint_redirects_velocity_without_adding_energy() {
    let config = PhysicsConfig {
        damping: 0.0,
        angular_damping: 0.0,
        substeps: 1,
        ..PhysicsConfig::default()
    };
    let mut world = observed_world(config);
    let anchor = world.spawn(BodyDesc::static_sphere(0.05).position([0.0, 30.0, 0.0]));
    let bob = world.spawn(
        BodyDesc::sphere(0.05)
            .position([1.0, 30.0, 0.0])
            .mass(1.0)
            .gravity_scale(0.0)
            .velocity([0.0, 0.0, 2.0]),
    );
    world.add_constraint(
        anchor,
        bob,
        ConstraintDesc::distance([0.0; 3], [0.0; 3], 1.0),
    );
    for _ in 0..400 {
        world.step(DT);
    }
    world.wait();
    let state = world.read_state(bob);
    let speed = speed_of(state.velocity);
    let radius = (state.position[0] * state.position[0]
        + (state.position[1] - 30.0) * (state.position[1] - 30.0)
        + state.position[2] * state.position[2])
        .sqrt();
    assert!(
        (radius - 1.0).abs() < 1e-3,
        "a rotating joint must keep its radius, got {radius}"
    );
    assert!(
        speed <= 2.2,
        "a rotating joint must not add energy to the orbit, speed {speed}"
    );
    assert!(
        speed >= 1.0,
        "a rotating joint must not drain the orbit, speed {speed}"
    );
}

#[test]
fn a_swinging_chain_holds_its_energy_budget() {
    let mut world = observed_world(gravity_config());
    let handles = swing_chain(&mut world, 5);
    let mut peak = 0.0f32;
    for frame in 0..1200 {
        world.step(DT);
        if frame % 10 != 9 {
            continue;
        }
        world.wait();
        for handle in &handles {
            peak = peak.max(speed_of(world.read_state(*handle).velocity));
        }
    }
    world.wait();
    assert!(
        peak < 20.0,
        "a swinging chain must not pump energy, peak speed {peak}"
    );
}
