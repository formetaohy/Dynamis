use super::common::{
    DT, asleep, gravity_config, observed_world, settle, settle_until, static_config,
};
use dynamis_abi::{COUNTER_ISLANDS, COUNTER_WOKE};
use dynamis_model::{BodyDesc, BodyHandle, SoftBodyDesc, SoftMaterial};
use dynamis_world::World;

fn ground(world: &mut World) {
    world.spawn(
        BodyDesc::cuboid([5.0, 0.5, 5.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
}

fn sleep_until_quiet(world: &mut World) {
    settle_until(world, 400, |world| asleep(world));
}

fn woken_bodies_over(world: &mut World, frames: usize) -> u32 {
    let mut woken = 0;
    for _ in 0..frames {
        world.step(DT);
        world.wait();
        woken += world.measured()[COUNTER_WOKE];
    }
    woken
}

#[test]
fn a_driven_body_pushes_the_sleeping_body_it_touches() {
    let mut world = observed_world(gravity_config());
    ground(&mut world);
    let ball = world.spawn(BodyDesc::sphere(0.5).position([0.0, 0.5, 0.0]));
    sleep_until_quiet(&mut world);
    assert!(world.read_state(ball).sleeping);

    world.spawn(
        BodyDesc::cuboid([0.5; 3])
            .position([-1.0, 0.5, 0.0])
            .kinematic(true)
            .velocity([1.0, 0.0, 0.0]),
    );
    settle(&mut world, 60);
    let state = world.read_state(ball);
    assert!(
        state.position[0] > 0.5,
        "a driven platform must carry the sleeping ball it touches, got {:?}",
        state.position
    );
}

#[test]
fn a_driven_body_wakes_and_lifts_a_sleeping_stack_as_one_island() {
    let mut world = observed_world(gravity_config());
    ground(&mut world);
    let platform = world.spawn(
        BodyDesc::cuboid([1.0, 0.1, 1.0])
            .mass(0.0)
            .position([0.0, 0.1, 0.0]),
    );
    let stack = (0..3)
        .map(|level| {
            world.spawn(BodyDesc::cuboid([0.4; 3]).position([0.0, 0.6 + level as f32 * 0.8, 0.0]))
        })
        .collect::<Vec<_>>();
    sleep_until_quiet(&mut world);
    assert!(stack.iter().all(|body| world.read_state(*body).sleeping));

    world.set_kinematic(platform, true);
    world.set_velocity(platform, [0.0, 0.15, 0.0]);
    settle(&mut world, 120);
    let heights = stack
        .iter()
        .map(|body| world.read_state(*body).position[1])
        .collect::<Vec<_>>();
    assert!(
        heights[0] > 0.8 && heights[1] > 1.6 && heights[2] > 2.4,
        "the whole sleeping island must ride the driven platform, got {heights:?}"
    );
    assert!(
        !world.read_state(stack[2]).sleeping,
        "the far end of the island must wake with the body the platform touches"
    );
}

#[test]
fn a_teleported_static_body_wakes_its_sleeping_neighbour() {
    let mut world = observed_world(gravity_config());
    ground(&mut world);
    let ball = world.spawn(BodyDesc::sphere(0.5).position([0.0, 0.5, 0.0]));
    sleep_until_quiet(&mut world);

    let wall = world.spawn(BodyDesc::static_sphere(0.5).position([5.0, 0.5, 0.0]));
    world.set_position(wall, [0.6, 0.5, 0.0]);
    assert!(
        woken_bodies_over(&mut world, 30) > 0,
        "a teleported static body must wake the sleeping body it overlaps"
    );
    let state = world.read_state(ball);
    assert!(
        state.position[0] < -0.3,
        "the awakened ball must leave the teleported body, got {:?}",
        state.position
    );
}

#[test]
fn a_teleported_driven_body_wakes_its_sleeping_neighbour() {
    let mut world = observed_world(gravity_config());
    ground(&mut world);
    let box_body = world.spawn(BodyDesc::cuboid([0.4; 3]).position([0.0, 0.4, 0.0]));
    sleep_until_quiet(&mut world);

    let pusher = world.spawn(
        BodyDesc::cuboid([0.4; 3])
            .position([4.0, 0.4, 0.0])
            .kinematic(true),
    );
    world.set_position(pusher, [0.6, 0.4, 0.0]);
    assert!(
        woken_bodies_over(&mut world, 30) > 0,
        "a teleported driven body must wake the sleeping body it overlaps"
    );
    let state = world.read_state(box_body);
    assert!(
        state.position[0] < -0.1,
        "the awakened box must leave the teleported body, got {:?}",
        state.position
    );
}

#[test]
fn waking_a_body_wakes_the_resting_island_it_belongs_to() {
    let mut world = observed_world(gravity_config());
    ground(&mut world);
    let lower = world.spawn(BodyDesc::sphere(0.5).position([0.0, 0.5, 0.0]));
    let upper = world.spawn(BodyDesc::sphere(0.5).position([0.0, 1.5, 0.0]));
    sleep_until_quiet(&mut world);
    assert!(
        world.read_state(lower).sleeping && world.read_state(upper).sleeping,
        "the pile must sleep before it is woken"
    );

    world.wake(lower);
    world.step(1.0 / 60.0);
    world.wait();
    assert!(
        !world.read_state(upper).sleeping,
        "waking one member must wake the resting island it belongs to"
    );
}

#[test]
fn a_soft_body_wakes_the_sleeping_body_it_pushes() {
    let mut world = observed_world(gravity_config());
    ground(&mut world);
    let box_body = world.spawn(BodyDesc::cuboid([0.3; 3]).position([0.0, 0.3, 0.0]));
    sleep_until_quiet(&mut world);
    assert!(world.read_state(box_body).sleeping);

    world.add_soft_body(
        SoftBodyDesc::lattice([3, 3, 3], 0.25, SoftMaterial::rigid())
            .radius(0.12)
            .position([-1.5, 0.3, 0.0])
            .velocity([4.0, 0.0, 0.0]),
    );
    assert!(
        woken_bodies_over(&mut world, 90) > 0,
        "a soft body must wake the sleeping body it pushes"
    );
    assert!(
        world.read_state(box_body).position[0] > 0.05,
        "the awakened box must be pushed along the impact direction, got {:?}",
        world.read_state(box_body).position
    );
}

#[test]
fn a_soft_body_wakes_the_sleeping_island_it_pushes() {
    let mut world = observed_world(gravity_config());
    ground(&mut world);
    let lower = world.spawn(
        BodyDesc::cuboid([0.3; 3])
            .position([0.0, 0.3, 0.0])
            .friction(0.6),
    );
    let upper = world.spawn(
        BodyDesc::cuboid([0.3; 3])
            .position([0.0, 0.9, 0.0])
            .friction(0.6),
    );
    sleep_until_quiet(&mut world);
    assert!(
        world.read_state(lower).sleeping && world.read_state(upper).sleeping,
        "a quiet stack must sleep before the soft impact"
    );

    world.add_soft_body(
        SoftBodyDesc::lattice([3, 3, 3], 0.25, SoftMaterial::rigid())
            .radius(0.12)
            .position([-1.5, 0.9, 0.0])
            .velocity([4.0, 0.0, 0.0]),
    );
    assert!(
        woken_bodies_over(&mut world, 90) >= 2,
        "a soft body must wake every member of the sleeping island it pushes"
    );
    assert!(
        world.read_state(upper).position[0] > 0.05,
        "the awakened island must move along the impact direction, got {:?}",
        world.read_state(upper).position
    );
}

#[test]
fn a_resting_island_keeps_every_member_asleep_together() {
    let mut world = observed_world(gravity_config());
    ground(&mut world);
    let bodies = (0..3)
        .map(|level| {
            world.spawn(
                BodyDesc::cuboid([0.4; 3])
                    .position([0.0, 0.4 + level as f32 * 0.8, 0.0])
                    .friction(0.6),
            )
        })
        .collect::<Vec<BodyHandle>>();
    let steps = settle_until(&mut world, 400, |world| asleep(world));
    assert!(steps < 400, "a resting stack must fall asleep");
    assert!(
        bodies.iter().all(|body| world.read_state(*body).sleeping),
        "every member of a quiet island must sleep"
    );
    settle(&mut world, 120);
    assert!(
        bodies.iter().all(|body| world.read_state(*body).sleeping),
        "a quiet island must stay asleep instead of waking itself"
    );
}

#[test]
fn a_host_patched_body_keeps_its_sleep_timer_reset() {
    let mut world = observed_world(gravity_config());
    ground(&mut world);
    let ball = world.spawn(
        BodyDesc::sphere(0.3)
            .position([0.0, 0.3, 0.0])
            .friction(0.6),
    );
    sleep_until_quiet(&mut world);
    assert!(world.read_state(ball).sleeping);

    for _ in 0..120 {
        world.set_velocity(ball, [0.1, 0.0, 0.0]);
        world.step(DT);
        world.wait();
    }
    let state = world.read_state(ball);
    assert!(
        !state.sleeping,
        "a body the host keeps writing must not fall asleep"
    );
    assert!(
        state.velocity[0] > 0.05,
        "a patched body must hold the velocity the host wrote, got {:?}",
        state.velocity
    );
}

#[test]
fn a_spawned_body_wakes_the_sleeping_body_it_overlaps() {
    let mut world = observed_world(static_config());
    let ball = world.spawn(BodyDesc::sphere(0.5));
    sleep_until_quiet(&mut world);
    assert!(world.read_state(ball).sleeping);

    world.spawn(BodyDesc::cuboid([0.4; 3]).position([0.3, 0.0, 0.0]));
    assert!(
        woken_bodies_over(&mut world, 30) > 0,
        "a spawned body must wake the sleeping body it overlaps"
    );
    assert!(
        world.read_state(ball).position[0] < -0.2,
        "the awakened ball must separate from the spawned body, got {:?}",
        world.read_state(ball).position
    );
}

#[test]
fn a_spawned_static_body_wakes_the_sleeping_body_it_overlaps() {
    let mut world = observed_world(static_config());
    let ball = world.spawn(BodyDesc::sphere(0.5));
    sleep_until_quiet(&mut world);
    assert!(world.read_state(ball).sleeping);

    world.spawn(BodyDesc::static_sphere(0.5).position([0.6, 0.0, 0.0]));
    assert!(
        woken_bodies_over(&mut world, 30) > 0,
        "a spawned static body must wake the sleeping body it overlaps"
    );
    assert!(
        world.read_state(ball).position[0] < -0.3,
        "the awakened ball must leave the spawned static body, got {:?}",
        world.read_state(ball).position
    );
}

#[test]
fn a_shared_support_holds_every_body_it_reaches_in_one_island() {
    let mut world = observed_world(static_config());
    world.spawn(BodyDesc::sphere(0.3).position([-0.5, 0.25, 0.0]));
    world.spawn(BodyDesc::sphere(0.3).position([0.5, 0.25, 0.0]));
    world.spawn(
        BodyDesc::cuboid([1.2, 0.25, 1.0])
            .position([0.0, -0.25, 0.0])
            .mass(4.0),
    );
    settle(&mut world, 4);
    assert_eq!(
        world.measured()[COUNTER_ISLANDS],
        1,
        "both spheres and the support they rest on must form one island",
    );
}

#[test]
fn a_coupled_body_wakes_when_its_support_leaves() {
    let mut world = observed_world(gravity_config());
    ground(&mut world);
    let left = world.spawn(BodyDesc::sphere(0.3).position([-0.5, 1.0, 0.0]));
    let right = world.spawn(BodyDesc::sphere(0.3).position([0.5, 1.0, 0.0]));
    let plank = world.spawn(
        BodyDesc::cuboid([1.2, 0.25, 1.0])
            .position([0.0, 0.25, 0.0])
            .mass(4.0)
            .friction(0.9),
    );
    sleep_until_quiet(&mut world);
    assert!(
        world.read_state(left).sleeping && world.read_state(right).sleeping,
        "both resting bodies must sleep before the support leaves"
    );

    world.set_position(plank, [50.0, 0.25, 0.0]);
    settle(&mut world, 120);
    assert!(
        world.read_state(left).position[1] < 0.5,
        "the body the support was taken from must fall, got {:?}",
        world.read_state(left).position
    );
    assert!(
        world.read_state(right).position[1] < 0.5,
        "every body coupled to the leaving support must fall, got {:?}",
        world.read_state(right).position
    );
}
