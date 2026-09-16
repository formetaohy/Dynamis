use super::common::{DT, gravity_config, new_world, observed_world, static_sphere_ground};
use dynamis_abi::{COUNTER_ACTIVE, COUNTER_CONTACTS, COUNTER_STEP};
use dynamis_model::{BodyDesc, SoftBodyDesc, SoftBodyHandle};
use dynamis_world::World;

fn hanging_net(world: &mut World) -> SoftBodyHandle {
    world.add_soft_body(SoftBodyDesc::net(
        vec![[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
        vec![[0, 1]],
    ))
}

#[test]
fn collecting_facts_never_opens_a_submission() {
    let mut world = observed_world(gravity_config());
    let _ground = static_sphere_ground(&mut world, 1.0);
    let _ball = world.spawn(BodyDesc::sphere(0.5).position([0.0, 1.4, 0.0]));
    world.step(DT);
    let base = world.submissions();
    for _ in 0..4 {
        world.poll();
        std::hint::black_box(world.collect_events());
        std::hint::black_box(world.collect_impacts());
        std::hint::black_box(world.collect_constraint_breaks());
    }
    assert_eq!(
        world.submissions(),
        base,
        "an arrival must deliver the facts it receives without opening a submission"
    );
}

#[test]
fn a_retirement_delivers_every_fact_of_the_step() {
    let mut world = new_world(gravity_config());
    let _ground = static_sphere_ground(&mut world, 1.0);
    let ball = world.spawn(BodyDesc::sphere(0.5).position([0.0, 1.4, 0.0]));
    world.observe_bodies(&[ball]);
    world.step(DT);
    assert_eq!(
        world.measured()[COUNTER_STEP],
        0,
        "a submitted step is not yet a fact"
    );
    world.drain_events();
    assert_eq!(
        world.measured()[COUNTER_STEP],
        1,
        "a retirement must seal the step it reads"
    );
    assert_eq!(
        world.measured()[COUNTER_ACTIVE],
        1,
        "a retirement must deliver the activity of the step it reads"
    );
    assert!(
        world.measured()[COUNTER_CONTACTS] > 0,
        "a retirement must deliver the contacts of the step it reads"
    );
    world.wait();
    let landed = world.read_state(ball);
    assert_eq!(
        landed.prev_position,
        [0.0, 1.4, 0.0],
        "a landing must bring the state of the step it landed"
    );
    assert_ne!(
        landed.position[1], 1.4,
        "a landing must bring the stepped pose rather than the spawn pose"
    );
}

#[test]
fn an_inspection_retires_the_step_it_reads() {
    let mut world = observed_world(gravity_config());
    let net = hanging_net(&mut world);
    world.step(DT);
    assert_eq!(world.measured()[COUNTER_STEP], 0);
    let particles = world.inspect_soft_particles(net);
    assert_eq!(
        particles.len(),
        2,
        "a soft inspection must return the particles of its own body"
    );
    assert_eq!(
        world.measured()[COUNTER_STEP],
        1,
        "an inspection must retire the step it reads"
    );
}

#[test]
fn a_snapshot_retires_the_facts_it_captures() {
    let mut world = observed_world(gravity_config());
    let _ball = world.spawn(BodyDesc::sphere(0.5).position([0.0, 1.0, 0.0]));
    world.step(DT);
    let snapshot = world.snapshot();
    assert_eq!(
        world.measured()[COUNTER_STEP],
        1,
        "a snapshot must retire the step it captures"
    );
    world.step(DT);
    world.restore(&snapshot);
    assert_eq!(
        world.measured()[COUNTER_STEP],
        0,
        "a restoration must retire the world it replaces before it restarts its measures"
    );
}
