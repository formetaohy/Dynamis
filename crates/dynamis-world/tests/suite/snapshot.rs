use super::common::{DT, flat_mesh_floor, gravity_config, new_world};
use dynamis_abi::{
    COUNTER_ACTIVE, COUNTER_ARCHIVED, COUNTER_CONTACTS, COUNTER_JOINTS, COUNTER_RESTING,
};
use dynamis_model::{BodyDesc, BodyHandle, ConstraintDesc, SoftBodyDesc, SoftBodyHandle};
use dynamis_world::{Snapshot, World};

const FRAMES: usize = 40;
const TAIL: usize = 12;
const BURST: usize = 200;

struct Scenario {
    bodies: Vec<BodyHandle>,
    soft: Vec<SoftBodyHandle>,
}

fn build(world: &mut World) -> Scenario {
    let mut bodies = vec![
        flat_mesh_floor(world),
        world.spawn(BodyDesc::sphere(0.2).position([0.0, 4.0, 0.0])),
    ];
    for index in 0..3 {
        bodies.push(
            world.spawn(
                BodyDesc::cuboid([0.4, 0.4, 0.4])
                    .position([0.05 * index as f32, 0.45 + index as f32 * 0.82, 0.0])
                    .friction(0.9),
            ),
        );
    }
    for index in 0..3 {
        let body = world.spawn(BodyDesc::sphere(0.25).position([2.0, 4.0, index as f32 * 0.6]));
        world.add_constraint(
            bodies[1],
            body,
            ConstraintDesc::ball([0.0; 3], [0.0; 3]).disable_collisions(false),
        );
        bodies.push(body);
    }
    let soft = vec![
        world.add_soft_body(SoftBodyDesc::net(vertices(), links()).position([-2.0, 2.0, 0.0])),
    ];
    Scenario { bodies, soft }
}

fn vertices() -> Vec<[f32; 3]> {
    vec![
        [-1.0, 0.0, -1.0],
        [1.0, 0.0, -1.0],
        [1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
        [-1.0, 1.0, -1.0],
        [1.0, 1.0, -1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ]
}

fn links() -> Vec<[u32; 2]> {
    vec![
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 0],
        [4, 5],
        [5, 6],
        [6, 7],
        [7, 4],
        [0, 4],
        [1, 5],
        [2, 6],
        [3, 7],
        [0, 2],
        [1, 3],
        [4, 6],
        [5, 7],
    ]
}

fn body_bits(world: &World, scenario: &Scenario) -> Vec<u32> {
    scenario
        .bodies
        .iter()
        .flat_map(|handle| {
            let body = world.read_state(*handle);
            let mut bits = Vec::new();
            bits.extend(body.position.map(f32::to_bits));
            bits.extend(body.orientation.map(f32::to_bits));
            bits.extend(body.velocity.map(f32::to_bits));
            bits.extend(body.angular_velocity.map(f32::to_bits));
            bits.push(u32::from(body.sleeping));
            bits
        })
        .collect()
}

fn soft_bits(world: &mut World, scenario: &Scenario) -> Vec<u32> {
    scenario
        .soft
        .iter()
        .flat_map(|handle| {
            world
                .inspect_soft_particles(*handle)
                .into_iter()
                .flat_map(|position| position.map(f32::to_bits))
                .collect::<Vec<_>>()
        })
        .collect()
}

fn state_bits(world: &mut World, scenario: &Scenario) -> Vec<u32> {
    let mut bits = body_bits(world, scenario);
    bits.extend(soft_bits(world, scenario));
    bits
}

fn world_bits(world: &mut World, scenario: &Scenario) -> Vec<u32> {
    let mut bits = state_bits(world, scenario);
    let measured = world.measured();
    bits.extend(
        [
            COUNTER_ACTIVE,
            COUNTER_CONTACTS,
            COUNTER_ARCHIVED,
            COUNTER_RESTING,
            COUNTER_JOINTS,
        ]
        .map(|slot| measured[slot]),
    );
    bits
}

fn settle(world: &mut World, frames: usize) -> Scenario {
    let scenario = build(world);
    advance(world, frames);
    scenario
}

fn advance(world: &mut World, frames: usize) {
    for _ in 0..frames {
        world.step(DT);
    }
    world.wait();
}

fn restore(world: &mut World, snapshot: &Snapshot) {
    world.restore(snapshot);
    world.wait();
}

#[test]
fn a_snapshot_alone_describes_the_world_it_captured() {
    let mut author = new_world(gravity_config());
    let scenario = settle(&mut author, FRAMES);
    let snapshot = author.snapshot();
    let mut copy = new_world(gravity_config());
    advance(&mut copy, 3);
    restore(&mut copy, &snapshot);
    assert_eq!(
        state_bits(&mut copy, &scenario),
        state_bits(&mut author, &scenario),
        "a snapshot must describe every frame of the world it captured"
    );
}

#[test]
fn a_restored_world_reproduces_the_frames_it_was_captured_from() {
    let mut world = new_world(gravity_config());
    let scenario = settle(&mut world, FRAMES);
    let snapshot = world.snapshot();
    let captured = state_bits(&mut world, &scenario);
    advance(&mut world, TAIL);
    let expected = world_bits(&mut world, &scenario);
    restore(&mut world, &snapshot);
    assert_eq!(
        state_bits(&mut world, &scenario),
        captured,
        "a restored world must stand exactly where it was captured"
    );
    advance(&mut world, TAIL);
    assert_eq!(
        world_bits(&mut world, &scenario),
        expected,
        "a restored world must replay the frames it was captured from"
    );
    restore(&mut world, &snapshot);
    advance(&mut world, TAIL);
    assert_eq!(
        world_bits(&mut world, &scenario),
        expected,
        "a snapshot must restore any number of times"
    );
}

#[test]
fn a_snapshot_replaces_the_scene_of_the_world_it_lands_in() {
    let mut author = new_world(gravity_config());
    let scenario = settle(&mut author, FRAMES);
    let snapshot = author.snapshot();
    advance(&mut author, TAIL);
    let expected = world_bits(&mut author, &scenario);

    let mut host = new_world(gravity_config());
    for index in 0..2 {
        host.spawn(BodyDesc::sphere(0.5).position([8.0, 8.0 + index as f32, 8.0]));
    }
    advance(&mut host, 3);
    assert_eq!(host.count(), 2, "the host world starts with its own scene");
    restore(&mut host, &snapshot);
    assert_eq!(
        host.count(),
        author.count(),
        "a snapshot installs its bodies"
    );
    assert_eq!(
        host.soft_body_count(),
        author.soft_body_count(),
        "a snapshot installs its soft bodies"
    );
    advance(&mut host, TAIL);
    assert_eq!(
        world_bits(&mut host, &scenario),
        expected,
        "a snapshot must replay its frames in any world it lands in"
    );
}

#[test]
fn a_restored_world_keeps_simulating_the_snapshot_that_grew_its_storage() {
    let mut author = new_world(gravity_config());
    let ball = author.spawn(BodyDesc::sphere(0.2).position([0.0, 6.0, 0.0]));
    let burst = (0..BURST)
        .map(|index| author.spawn(BodyDesc::sphere(0.2).position([40.0 + index as f32, 6.0, 0.0])))
        .collect::<Vec<_>>();
    advance(&mut author, 4);
    for body in burst {
        author.remove(body);
    }
    advance(&mut author, 3);
    let snapshot = author.snapshot();

    let mut host = new_world(gravity_config());
    host.spawn(BodyDesc::sphere(0.2).position([-40.0, 6.0, 0.0]));
    advance(&mut host, 2);
    restore(&mut host, &snapshot);

    let captured = host.read_state(ball).position;
    advance(&mut host, TAIL);
    let replayed = host.read_state(ball).position;
    assert!(
        replayed[1] < captured[1] - 0.1,
        "a restored world must keep simulating the storage it installed, {captured:?} -> {replayed:?}"
    );
}

#[test]
fn a_restored_world_requires_a_fresh_observation() {
    let mut world = new_world(gravity_config());
    let scenario = settle(&mut world, FRAMES);
    let snapshot = world.snapshot();
    let captured = world
        .read_state(scenario.bodies[1])
        .position
        .map(f32::to_bits);
    advance(&mut world, TAIL);
    world.restore(&snapshot);
    assert!(
        world.try_state(scenario.bodies[1]).is_none(),
        "a restored world must not answer with the frames it replaced"
    );
    world.wait();
    assert_eq!(
        world
            .read_state(scenario.bodies[1])
            .position
            .map(f32::to_bits),
        captured,
        "a fresh observation must report the captured frame"
    );
}
