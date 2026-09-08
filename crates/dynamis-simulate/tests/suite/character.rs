use super::common::{DT, sim};
use dynamis_model::BodyDesc;
use dynamis_simulate::{CharacterDesc, Simulation};

fn walk_and_settle(
    world: &mut Simulation,
    character: &mut dynamis_simulate::Character,
    frames: usize,
    direction: [f32; 3],
) {
    for _ in 0..frames {
        world.step(DT);
        world.character_step(character, DT, direction, false);
        world.wait();
    }
}

#[test]
fn character_walks_on_flat_ground() {
    let mut world = sim(8, super::common::gravity_config());
    world.spawn(
        BodyDesc::cuboid([20.0, 0.5, 20.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let mut character = world.spawn_character([0.0, 1.0, 0.0], CharacterDesc::default());
    walk_and_settle(&mut world, &mut character, 30, [1.0, 0.0, 0.0]);
    assert!(character.grounded(), "character must stay grounded");
    assert!(
        character.position()[0] > 0.5,
        "character must advance along its move direction, got x={}",
        character.position()[0]
    );
    assert!(
        (character.position()[1] - 1.0).abs() < 0.2,
        "character height must stay constant, got y={}",
        character.position()[1]
    );
}

#[test]
fn character_climbs_step() {
    let mut world = sim(8, super::common::gravity_config());
    world.spawn(
        BodyDesc::cuboid([20.0, 0.5, 20.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    world.spawn(
        BodyDesc::cuboid([1.0, 0.2, 4.0])
            .mass(0.0)
            .position([2.0, 0.1, 0.0]),
    );
    let mut character = world.spawn_character(
        [0.0, 1.0, 0.0],
        CharacterDesc {
            step_height: 0.3,
            ..Default::default()
        },
    );
    walk_and_settle(&mut world, &mut character, 40, [1.0, 0.0, 0.0]);
    walk_and_settle(&mut world, &mut character, 30, [0.0, 0.0, 0.0]);
    assert!(
        character.position()[0] > 2.2,
        "character must cross the step, got x={}",
        character.position()[0]
    );
    assert!(
        (character.position()[1] - 1.2).abs() < 0.12,
        "character must stand on top of the step, got y={}",
        character.position()[1]
    );
    assert!(
        character.grounded() || character.position()[1] > 1.1,
        "character must rest on the step surface"
    );
}

#[test]
fn character_blocked_by_tall_wall() {
    let mut world = sim(8, super::common::gravity_config());
    world.spawn(
        BodyDesc::cuboid([20.0, 0.5, 20.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    world.spawn(
        BodyDesc::cuboid([0.5, 3.0, 8.0])
            .mass(0.0)
            .position([2.0, 1.5, 0.0]),
    );
    let mut character = world.spawn_character([0.0, 1.0, 0.0], CharacterDesc::default());
    walk_and_settle(&mut world, &mut character, 60, [1.0, 0.0, 0.0]);
    assert!(
        character.position()[0] < 2.0 - 0.3,
        "character must stop at the wall, got x={}",
        character.position()[0]
    );
}

#[test]
fn character_lands_after_jump() {
    let mut world = sim(8, super::common::gravity_config());
    world.spawn(
        BodyDesc::cuboid([20.0, 0.5, 20.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let mut character = world.spawn_character([0.0, 1.0, 0.0], CharacterDesc::default());
    world.step(DT);
    world.character_step(&mut character, DT, [0.0, 0.0, 0.0], true);
    world.wait();
    let mut apex = character.position()[1];
    for _ in 0..10 {
        world.step(DT);
        world.character_step(&mut character, DT, [0.0, 0.0, 0.0], false);
        world.wait();
        apex = apex.max(character.position()[1]);
    }
    assert!(apex > 1.35, "jump must lift the character, got apex {apex}");
    for _ in 0..60 {
        world.step(DT);
        world.character_step(&mut character, DT, [0.0, 0.0, 0.0], false);
        world.wait();
    }
    assert!(
        character.grounded(),
        "character must land back on the ground"
    );
}

#[test]
fn character_pushes_dynamic_box() {
    let mut world = sim(8, super::common::gravity_config());
    world.spawn(
        BodyDesc::cuboid([20.0, 0.5, 20.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let box_body = world.spawn(
        BodyDesc::cuboid([0.5, 0.75, 0.5])
            .position([1.6, 0.75, 0.0])
            .friction(0.6),
    );
    let mut character = world.spawn_character([0.0, 1.0, 0.0], CharacterDesc::default());
    walk_and_settle(&mut world, &mut character, 100, [1.0, 0.0, 0.0]);
    let moved = world.read_state(box_body).position[0];
    assert!(
        moved > 1.75,
        "character must push the box forward, box at x={moved}"
    );
}

#[test]
fn character_climbs_walkable_slope() {
    let mut world = sim(8, super::common::gravity_config());
    let angle = 20.0_f32.to_radians();
    let tilt = [0.0, 0.0, (angle * 0.5).sin(), (angle * 0.5).cos()];
    world.spawn(
        BodyDesc::cuboid([10.0, 0.5, 10.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let rise = 4.0 * angle.tan();
    world.spawn(
        BodyDesc::cuboid([4.0, 0.3, 4.0])
            .mass(0.0)
            .position([2.0, rise * 0.5 - 0.15, 0.0])
            .orientation(tilt),
    );
    let mut character = world.spawn_character(
        [0.0, 1.0, 0.0],
        CharacterDesc {
            slope_limit: 30.0_f32.to_radians(),
            ..Default::default()
        },
    );
    walk_and_settle(&mut world, &mut character, 60, [1.0, 0.0, 0.0]);
    let position = character.position();
    assert!(
        position[0] > 2.5,
        "character must advance up the ramp, got x={}",
        position[0]
    );
    assert!(
        position[1] > 1.4,
        "character must gain height on the ramp, got y={}",
        position[1]
    );
}
