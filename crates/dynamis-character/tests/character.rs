mod common;

use common::{DT, gravity_config, new_world};
use dynamis_character::{Character, CharacterDesc};
use dynamis_model::{BodyDesc, CharacterState, ColliderDesc, Shape};
use dynamis_world::World;

fn ground(world: &mut World) {
    world.spawn(
        BodyDesc::cuboid([20.0, 0.5, 20.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
}

fn walk(world: &mut World, character: &Character, frames: usize, direction: [f32; 3]) {
    for _ in 0..frames {
        character.drive(world, direction, false);
        world.step(DT);
    }
}

fn settled(world: &mut World, character: &Character) -> CharacterState {
    world.wait();
    character.inspect_state(world)
}

#[test]
fn character_walks_on_flat_ground() {
    let mut world = new_world(gravity_config());
    ground(&mut world);
    let character = Character::spawn(&mut world, [0.0, 1.0, 0.0], CharacterDesc::default());
    walk(&mut world, &character, 30, [1.0, 0.0, 0.0]);
    let state = settled(&mut world, &character);
    assert!(state.grounded, "character must stay grounded");
    assert!(
        state.position[0] > 0.5,
        "character must advance along its move direction, got x={}",
        state.position[0]
    );
    assert!(
        (state.position[1] - 1.0).abs() < 0.2,
        "character height must stay constant, got y={}",
        state.position[1]
    );
}

#[test]
fn character_climbs_step() {
    let mut world = new_world(gravity_config());
    ground(&mut world);
    world.spawn(
        BodyDesc::cuboid([1.0, 0.2, 4.0])
            .mass(0.0)
            .position([2.0, 0.1, 0.0]),
    );
    let character = Character::spawn(
        &mut world,
        [0.0, 1.0, 0.0],
        CharacterDesc {
            step_height: 0.3,
            ..Default::default()
        },
    );
    walk(&mut world, &character, 40, [1.0, 0.0, 0.0]);
    walk(&mut world, &character, 30, [0.0, 0.0, 0.0]);
    let state = settled(&mut world, &character);
    assert!(
        state.position[0] > 2.2,
        "character must cross the step, got x={}",
        state.position[0]
    );
    assert!(
        (state.position[1] - 1.2).abs() < 0.12,
        "character must stand on top of the step, got y={}",
        state.position[1]
    );
    assert!(
        state.grounded || state.position[1] > 1.1,
        "character must rest on the step surface"
    );
}

#[test]
fn character_blocked_by_tall_wall() {
    let mut world = new_world(gravity_config());
    ground(&mut world);
    world.spawn(
        BodyDesc::cuboid([0.5, 3.0, 8.0])
            .mass(0.0)
            .position([2.0, 1.5, 0.0]),
    );
    let character = Character::spawn(&mut world, [0.0, 1.0, 0.0], CharacterDesc::default());
    walk(&mut world, &character, 60, [1.0, 0.0, 0.0]);
    let state = settled(&mut world, &character);
    assert!(
        state.position[0] < 2.0 - 0.3,
        "character must stop at the wall, got x={}",
        state.position[0]
    );
}

#[test]
fn character_lands_after_jump() {
    let mut world = new_world(gravity_config());
    ground(&mut world);
    let character = Character::spawn(&mut world, [0.0, 1.0, 0.0], CharacterDesc::default());
    character.drive(&mut world, [0.0, 0.0, 0.0], true);
    world.step(DT);
    let mut apex = character.inspect_state(&mut world).position[1];
    for _ in 0..10 {
        character.drive(&mut world, [0.0, 0.0, 0.0], false);
        world.step(DT);
        apex = apex.max(character.inspect_state(&mut world).position[1]);
    }
    assert!(apex > 1.35, "jump must lift the character, got apex {apex}");
    for _ in 0..60 {
        character.drive(&mut world, [0.0, 0.0, 0.0], false);
        world.step(DT);
    }
    let state = settled(&mut world, &character);
    assert!(state.grounded, "character must land back on the ground");
}

#[test]
fn character_pushes_dynamic_box() {
    let mut world = new_world(gravity_config());
    ground(&mut world);
    let box_body = world.spawn(
        BodyDesc::cuboid([0.5, 0.75, 0.5])
            .position([1.6, 0.75, 0.0])
            .friction(0.6),
    );
    let character = Character::spawn(&mut world, [0.0, 1.0, 0.0], CharacterDesc::default());
    walk(&mut world, &character, 100, [1.0, 0.0, 0.0]);
    world.wait();
    let moved = world.read_state(box_body).position[0];
    assert!(
        moved > 1.75,
        "character must push the box forward, box at x={moved}"
    );
}

#[test]
fn character_climbs_walkable_slope() {
    let mut world = new_world(gravity_config());
    let angle = 20.0_f32.to_radians();
    let tilt = [0.0, 0.0, (angle * 0.5).sin(), (angle * 0.5).cos()];
    ground(&mut world);
    let rise = 4.0 * angle.tan();
    world.spawn(
        BodyDesc::cuboid([4.0, 0.3, 4.0])
            .mass(0.0)
            .position([2.0, rise * 0.5 - 0.15, 0.0])
            .orientation(tilt),
    );
    let character = Character::spawn(
        &mut world,
        [0.0, 1.0, 0.0],
        CharacterDesc {
            slope_limit: 30.0_f32.to_radians(),
            ..Default::default()
        },
    );
    walk(&mut world, &character, 60, [1.0, 0.0, 0.0]);
    let state = settled(&mut world, &character);
    assert!(
        state.position[0] > 2.5,
        "character must advance up the ramp, got x={}",
        state.position[0]
    );
    assert!(
        state.position[1] > 1.4,
        "character must gain height on the ramp, got y={}",
        state.position[1]
    );
}

#[test]
fn character_control_rides_the_world_step() {
    let mut world = new_world(gravity_config());
    ground(&mut world);
    let character = Character::spawn(&mut world, [0.0, 1.0, 0.0], CharacterDesc::default());
    walk(&mut world, &character, 10, [1.0, 0.0, 0.0]);
    assert!(character.inspect_state(&mut world).grounded);
    let base = world.submissions();
    for _ in 0..10 {
        character.drive(&mut world, [1.0, 0.0, 0.0], false);
        world.step(DT);
    }
    assert_eq!(
        world.submissions() - base,
        10,
        "driving a character must ride the world step instead of opening a submission"
    );
    assert!(
        character.inspect_state(&mut world).grounded,
        "a driven character must stay grounded on flat ground"
    );
}

#[test]
fn character_grounds_on_a_plane_floor() {
    let mut world = new_world(gravity_config());
    world.spawn(BodyDesc::new(ColliderDesc::new(Shape::plane())).mass(0.0));
    let character = Character::spawn(&mut world, [0.0, 1.0, 0.0], CharacterDesc::default());
    walk(&mut world, &character, 40, [1.0, 0.0, 0.0]);
    let state = settled(&mut world, &character);
    assert!(state.grounded, "a plane floor must ground the character");
    assert!(
        (state.position[1] - 1.0).abs() < 0.2,
        "a plane floor must hold the character height, got y={}",
        state.position[1]
    );
}

#[test]
fn character_grounds_on_a_mesh_floor() {
    let mut world = new_world(gravity_config());
    let floor = world.add_mesh(
        &[
            [-20.0, 0.0, -20.0],
            [20.0, 0.0, -20.0],
            [20.0, 0.0, 20.0],
            [-20.0, 0.0, 20.0],
        ],
        &[[0, 1, 2], [0, 2, 3]],
        None,
    );
    world.spawn(BodyDesc::new(ColliderDesc::new(Shape::mesh(floor))).mass(0.0));
    let character = Character::spawn(&mut world, [0.0, 1.0, 0.0], CharacterDesc::default());
    walk(&mut world, &character, 40, [1.0, 0.0, 0.0]);
    let state = settled(&mut world, &character);
    assert!(state.grounded, "a mesh floor must ground the character");
    assert!(
        (state.position[1] - 1.0).abs() < 0.2,
        "a mesh floor must hold the character height, got y={}",
        state.position[1]
    );
}

#[test]
fn character_state_is_a_published_device_fact() {
    let mut world = new_world(gravity_config());
    ground(&mut world);
    let character = Character::spawn(&mut world, [0.0, 1.0, 0.0], CharacterDesc::default());
    assert!(
        character.try_state(&mut world).is_none(),
        "a character state that no run has published must not be invented by the host"
    );
    let spawned = character.inspect_state(&mut world);
    assert_eq!(
        spawned.position,
        [0.0, 1.0, 0.0],
        "a spawned character must carry the state the host seeded it with"
    );
    assert!(spawned.grounded, "a spawned character starts grounded");
    walk(&mut world, &character, 20, [1.0, 0.0, 0.0]);
    world.wait();
    let observed = character
        .try_state(&mut world)
        .expect("wait() must publish the state of a watched character");
    assert_eq!(observed.value, character.inspect_state(&mut world));
    assert!(
        observed.value.position[0] > 0.0,
        "the published state must follow the simulated character"
    );
}

#[test]
fn a_published_state_follows_the_slot_it_belongs_to() {
    let mut world = new_world(gravity_config());
    ground(&mut world);
    let first = Character::spawn(&mut world, [-8.0, 1.0, 0.0], CharacterDesc::default());
    let second = Character::spawn(&mut world, [0.0, 1.0, 0.0], CharacterDesc::default());
    assert!(
        second.try_state(&mut world).is_none(),
        "a character state is published only after the host subscribes to it"
    );
    walk(&mut world, &second, 20, [1.0, 0.0, 0.0]);
    world.remove_character(first.handle());
    walk(&mut world, &second, 20, [1.0, 0.0, 0.0]);
    world.wait();
    let observed = second
        .try_state(&mut world)
        .expect("wait() must publish the surviving character");
    assert_eq!(observed.value, second.inspect_state(&mut world));
    assert!(
        observed.value.position[0] > 1.0,
        "the surviving character must publish its own slot, got x={}",
        observed.value.position[0]
    );
}

#[test]
fn retiring_a_character_leaves_the_resting_world_asleep() {
    let mut world = new_world(gravity_config());
    ground(&mut world);
    let ball = world.spawn(
        BodyDesc::sphere(0.5)
            .position([5.0, 0.5, 0.0])
            .friction(0.9),
    );
    for _ in 0..120 {
        world.step(DT);
    }
    world.wait();
    assert!(
        world.read_state(ball).sleeping,
        "the ball must settle before the character churn"
    );
    let character = Character::spawn(&mut world, [0.0, 1.0, 0.0], CharacterDesc::default());
    walk(&mut world, &character, 4, [1.0, 0.0, 0.0]);
    world.remove_character(character.handle());
    for _ in 0..120 {
        world.step(DT);
    }
    world.wait();
    assert!(
        world.read_state(ball).sleeping,
        "a retired character must not keep the simulation awake"
    );
}

#[test]
fn character_drives_its_kinematic_body() {
    let mut world = new_world(gravity_config());
    ground(&mut world);
    let character = Character::spawn(&mut world, [0.0, 1.0, 0.0], CharacterDesc::default());
    walk(&mut world, &character, 10, [1.0, 0.0, 0.0]);
    let state = settled(&mut world, &character);
    let body = world.read_state(character.body(&world));
    assert!(
        (body.position[0] - state.position[0]).abs() < 0.1,
        "the character body must track the character, body x={} character x={}",
        body.position[0],
        state.position[0]
    );
    assert!(
        body.position[1] > 0.0,
        "the kinematic body must stay above the floor"
    );
}

#[test]
fn removing_a_character_keeps_the_others_running() {
    let mut world = new_world(gravity_config());
    ground(&mut world);
    let first = Character::spawn(&mut world, [-4.0, 1.0, 0.0], CharacterDesc::default());
    let second = Character::spawn(&mut world, [0.0, 1.0, 0.0], CharacterDesc::default());
    walk(&mut world, &second, 10, [1.0, 0.0, 0.0]);
    let before = second.inspect_state(&mut world);
    world.remove_character(first.handle());
    assert_eq!(world.character_count(), 1);
    walk(&mut world, &second, 30, [1.0, 0.0, 0.0]);
    let after = settled(&mut world, &second);
    assert!(
        after.grounded,
        "the surviving character must stay grounded after a removal"
    );
    assert!(
        after.position[0] > before.position[0],
        "the surviving character must keep walking after a removal"
    );
    assert!(
        (after.position[1] - before.position[1]).abs() < 0.2,
        "the surviving character must keep its height after a removal"
    );
}
