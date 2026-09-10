use super::common::{DT, gravity_config, sim, static_config};
use dynamis_layout::{
    COUNTER_ACTIVE, COUNTER_CONTACTS, COUNTER_ENTRIES, COUNTER_PAIRS, COUNTER_RESTING,
    COUNTER_SLEPT, COUNTER_WOKE,
};
use dynamis_model::{
    BodyDesc, BodyHandle, ColliderDesc, ContactEventKind, ContactEventMode, PhysicsConfig, Shape,
};

fn settle(world: &mut dynamis_simulate::Simulation, steps: usize) {
    for _ in 0..steps {
        world.step(DT);
        world.wait();
    }
}

fn rest_scene() -> (dynamis_simulate::Simulation, BodyHandle) {
    let mut world = sim(64, gravity_config());
    world.spawn(
        BodyDesc::cuboid([5.0, 0.5, 5.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let bodies = (0..16)
        .map(|index| {
            let x = (index % 4) as f32 * 0.85 - 1.3;
            let z = (index / 4) as f32 * 0.85 - 1.3;
            world.spawn(BodyDesc::sphere(0.4).position([x, 0.4, z]).friction(0.6))
        })
        .collect::<Vec<_>>();
    (world, bodies[0])
}

#[test]
fn a_pile_at_rest_leaves_the_simulation_domain() {
    let (mut world, _) = rest_scene();
    settle(&mut world, 20);
    assert!(
        world.measured()[COUNTER_PAIRS] > 0,
        "an unsettled pile must generate pairs"
    );
    settle(&mut world, 400);
    let settled = world.measured();
    assert_eq!(
        settled[COUNTER_ACTIVE], 0,
        "every body must be asleep once the pile settles"
    );
    assert_eq!(
        settled[COUNTER_ENTRIES], 0,
        "a sleeping world must not maintain a broadphase grid"
    );
    assert_eq!(
        settled[COUNTER_PAIRS], 0,
        "a sleeping world must not generate pairs"
    );
    assert_eq!(
        settled[COUNTER_CONTACTS], 0,
        "a sleeping world must not hold live contacts"
    );
}

#[test]
fn resting_contacts_survive_sleep_and_recycle_their_slots() {
    let (mut world, bottom) = rest_scene();
    settle(&mut world, 400);
    let resting = world.measured()[COUNTER_RESTING];
    assert!(
        resting > 0,
        "a sleeping pile must keep its contacts in the resting store"
    );
    assert_eq!(
        world.contact_manifolds().len(),
        resting as usize,
        "the public contact list must report the resting contacts"
    );

    world.wake(bottom);
    world.step(DT);
    world.wait();
    assert!(
        world.measured()[COUNTER_WOKE] > 0,
        "the wake must be counted"
    );
    assert!(
        world.measured()[COUNTER_PAIRS] > 0,
        "a woken world must generate pairs again"
    );
    for _ in 0..3 {
        for handle in world.bodies().to_vec() {
            world.wake(handle);
        }
        settle(&mut world, 400);
    }
    assert!(
        world.measured()[COUNTER_RESTING] <= resting + 16,
        "sleeping again must recycle resting slots instead of appending"
    );
}

#[test]
fn sleeping_a_pair_never_ends_its_contact() {
    let mut world = sim(8, gravity_config());
    world.spawn(
        BodyDesc::cuboid([5.0, 0.5, 5.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let ball = world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::sphere(0.5)).events(ContactEventMode::BeginEnd))
            .position([0.0, 2.0, 0.0]),
    );
    let mut began = 0;
    for _ in 0..300 {
        world.step(DT);
        world.wait();
        for event in world.drain_events() {
            match event.kind {
                ContactEventKind::Begin => began += 1,
                ContactEventKind::End => {
                    panic!("a resting pair must not report the end of its contact")
                }
                ContactEventKind::Persist => {}
            }
        }
        if world.read_state(ball).sleeping {
            break;
        }
    }
    assert_eq!(began, 1, "the landing must report exactly one begin");
}

#[test]
fn sleep_and_wake_transitions_are_counted_once() {
    let mut world = sim(8, static_config());
    let ball = world.spawn(BodyDesc::sphere(0.5));
    settle(&mut world, 40);
    assert!(world.read_state(ball).sleeping);
    assert!(
        world.measured()[COUNTER_SLEPT] == 0,
        "a body that is already asleep must not be counted again"
    );
    world.wake(ball);
    world.step(DT);
    world.wait();
    assert_eq!(
        world.measured()[COUNTER_WOKE],
        1,
        "the wake command must count exactly one transition"
    );
}

#[test]
fn static_pairs_never_reach_the_pair_stream() {
    let mut world = sim(64, static_config());
    assert!(PhysicsConfig::default().gravity[1] < 0.0);
    for index in 0..32 {
        world.spawn(BodyDesc::static_sphere(6.0).position([index as f32 * 0.5, 0.0, 0.0]));
    }
    world.step(DT);
    world.wait();
    assert_eq!(
        world.measured()[COUNTER_PAIRS],
        0,
        "two static colliders must never enter the pair stream"
    );
}
