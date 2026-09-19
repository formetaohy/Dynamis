mod common;

use common::{DT, gravity_config, new_world};
use dynamis_character::{Character, CharacterDesc};
use dynamis_model::{BodyDesc, BodyHandle, CharacterState};
use dynamis_world::World;

fn platform(world: &mut World, velocity: [f32; 3]) -> BodyHandle {
    world.spawn(
        BodyDesc::cuboid([6.0, 0.25, 6.0])
            .position([0.0, 0.0, 0.0])
            .kinematic(true)
            .velocity(velocity),
    )
}

fn floor(world: &mut World) -> BodyHandle {
    world.spawn(
        BodyDesc::cuboid([20.0, 0.5, 20.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    )
}

fn settle(world: &mut World, character: &Character, steps: usize) -> CharacterState {
    drive(world, character, steps, [0.0; 3])
}

fn drive(
    world: &mut World,
    character: &Character,
    steps: usize,
    direction: [f32; 3],
) -> CharacterState {
    for _ in 0..steps {
        character.drive(world, direction, false);
        world.step(DT);
    }
    world.wait();
    character.inspect_state(world)
}

#[test]
fn a_character_rides_the_platform_it_stands_on() {
    let mut world = new_world(gravity_config());
    let platform = platform(&mut world, [2.0, 0.0, 0.0]);
    let character = Character::spawn(&mut world, [0.0, 1.5, 0.0], CharacterDesc::default());
    let landed = settle(&mut world, &character, 40);
    assert!(landed.grounded, "the character must land on the platform");
    assert_eq!(
        landed.support,
        Some(platform),
        "a character standing on the platform must name it as its support"
    );
    let travelled = world.read_state(platform).position[0];
    let after = settle(&mut world, &character, 90);
    let carried = after.position[0] - landed.position[0];
    assert!(
        carried > 1.5,
        "a character on a sliding platform must be carried along, got {carried}"
    );
    assert!(
        (carried - (world.read_state(platform).position[0] - travelled)).abs() < 0.05,
        "the character must be carried as far as the platform travels, got {carried} against {}",
        world.read_state(platform).position[0] - travelled,
    );
    assert!(
        (after.position[1] - landed.position[1]).abs() < 0.05,
        "a carried character must keep its height, got {} against {}",
        after.position[1],
        landed.position[1],
    );
    assert_eq!(after.support, Some(platform));
}

#[test]
fn a_character_that_lands_on_a_moving_platform_is_carried_by_it() {
    let mut world = new_world(gravity_config());
    let platform = platform(&mut world, [1.5, 0.0, 0.0]);
    let character = Character::spawn(&mut world, [0.0, 3.0, 0.0], CharacterDesc::default());
    let landed = settle(&mut world, &character, 60);
    assert!(
        landed.grounded,
        "the character must land on the moving platform"
    );
    assert_eq!(landed.support, Some(platform));
    let offset = landed.position[0] - world.read_state(platform).position[0];
    let after = settle(&mut world, &character, 60);
    let drift = (after.position[0] - world.read_state(platform).position[0]) - offset;
    assert!(
        drift.abs() < 0.05,
        "a character that landed on a platform must keep the place it landed on, drifted {drift}"
    );
}

#[test]
fn a_rising_platform_lifts_the_character_it_carries() {
    let mut world = new_world(gravity_config());
    let platform = platform(&mut world, [0.0, 1.0, 0.0]);
    let character = Character::spawn(&mut world, [0.0, 1.2, 0.0], CharacterDesc::default());
    let landed = settle(&mut world, &character, 20);
    assert!(landed.grounded, "the character must land on the platform");
    let before = world.read_state(platform).position[1];
    let after = settle(&mut world, &character, 90);
    let lifted = after.position[1] - landed.position[1];
    assert!(
        lifted > 1.0,
        "a rising platform must lift the character it carries, got {lifted}"
    );
    assert!(
        (lifted - (world.read_state(platform).position[1] - before)).abs() < 0.05,
        "the character must rise as far as the platform, got {lifted} against {}",
        world.read_state(platform).position[1] - before,
    );
    assert!(
        after.grounded,
        "a character on a rising platform must stay grounded"
    );
}

#[test]
fn a_descending_platform_keeps_the_character_on_it() {
    let mut world = new_world(gravity_config());
    let platform = platform(&mut world, [0.0, -0.6, 0.0]);
    let character = Character::spawn(&mut world, [0.0, 1.2, 0.0], CharacterDesc::default());
    let landed = settle(&mut world, &character, 20);
    assert!(landed.grounded, "the character must land on the platform");
    let gap = landed.position[1] - world.read_state(platform).position[1];
    let after = settle(&mut world, &character, 90);
    let movement = landed.position[1] - after.position[1];
    assert!(
        movement > 0.6,
        "a descending platform must carry the character down with it, got {movement}"
    );
    assert!(
        ((after.position[1] - world.read_state(platform).position[1]) - gap).abs() < 0.05,
        "a character must keep its footing on a descending platform, got {}",
        after.position[1] - world.read_state(platform).position[1],
    );
    assert!(
        after.grounded,
        "a character on a descending platform must stay grounded"
    );
}

#[test]
fn a_turning_platform_carries_the_character_around_it() {
    let mut world = new_world(gravity_config());
    let platform = world.spawn(
        BodyDesc::cuboid([6.0, 0.25, 6.0])
            .position([0.0, 0.0, 0.0])
            .kinematic(true)
            .angular_velocity([0.0, 0.5, 0.0]),
    );
    let character = Character::spawn(&mut world, [3.0, 1.2, 0.0], CharacterDesc::default());
    let landed = settle(&mut world, &character, 40);
    assert!(
        landed.grounded,
        "the character must land on the turning platform"
    );
    let after = settle(&mut world, &character, 60);
    let turned =
        after.position[2].atan2(after.position[0]) - landed.position[2].atan2(landed.position[0]);
    assert!(
        (turned.abs() - 0.5).abs() < 0.1,
        "a turning platform must carry the character around it, turned {turned} rad"
    );
    let radius = after.position[0].hypot(after.position[2]);
    assert!(
        (radius - 3.0).abs() < 0.1,
        "a carried character must keep its distance from the axis, got {radius}"
    );
    assert_eq!(after.support, Some(platform));
}

#[test]
fn a_character_on_still_ground_names_it_and_stays_where_it_stands() {
    let mut world = new_world(gravity_config());
    let ground = floor(&mut world);
    let character = Character::spawn(&mut world, [0.0, 1.0, 0.0], CharacterDesc::default());
    let landed = settle(&mut world, &character, 30);
    assert!(landed.grounded, "the character must land on the ground");
    assert_eq!(
        landed.support,
        Some(ground),
        "a character must name the ground it stands on"
    );
    let after = settle(&mut world, &character, 30);
    assert!(
        after
            .position
            .iter()
            .zip(landed.position)
            .all(|(after, landed)| (after - landed).abs() < 1.0e-3),
        "a character on still ground must stay where it stands, got {:?} against {:?}",
        after.position,
        landed.position,
    );
}

#[test]
fn a_jump_leaves_the_support_and_lands_on_it_again() {
    let mut world = new_world(gravity_config());
    let platform = platform(&mut world, [1.0, 0.0, 0.0]);
    let character = Character::spawn(&mut world, [0.0, 1.2, 0.0], CharacterDesc::default());
    let landed = settle(&mut world, &character, 30);
    assert_eq!(landed.support, Some(platform));

    character.drive(&mut world, [0.0; 3], true);
    world.step(DT);
    world.wait();
    let risen = character.inspect_state(&mut world);
    assert!(
        !risen.grounded && risen.support.is_none(),
        "a character that jumped must carry no support, got {risen:?}"
    );
    let after = settle(&mut world, &character, 60);
    assert!(
        after.grounded && after.support == Some(platform),
        "a character must land on the support it jumped from, got {after:?}"
    );
}

#[test]
fn a_removed_support_stops_carrying_the_character() {
    let mut world = new_world(gravity_config());
    let platform = platform(&mut world, [2.0, 0.0, 0.0]);
    let character = Character::spawn(&mut world, [0.0, 1.2, 0.0], CharacterDesc::default());
    let landed = settle(&mut world, &character, 30);
    assert_eq!(landed.support, Some(platform));
    world.remove(platform);
    let falling = settle(&mut world, &character, 20);
    assert!(
        falling.support.is_none(),
        "a character whose support the world retired must name no support, got {falling:?}"
    );
    assert!(
        falling.position[1] < landed.position[1] - 0.1,
        "a character whose support the world retired must fall, got {} against {}",
        falling.position[1],
        landed.position[1],
    );
}

#[test]
fn a_placement_clears_the_support_the_character_left() {
    let mut world = new_world(gravity_config());
    let platform = platform(&mut world, [1.0, 0.0, 0.0]);
    let ground = floor(&mut world);
    let character = Character::spawn(&mut world, [0.0, 1.2, 0.0], CharacterDesc::default());
    let landed = settle(&mut world, &character, 30);
    assert_eq!(landed.support, Some(platform));

    character.place(&mut world, [-10.0, 1.0, 0.0]);
    world.step(DT);
    world.wait();
    let placed = character.inspect_state(&mut world);
    assert!(
        placed.support.is_none(),
        "a placed character must answer no support until its sweeps answer one, got {placed:?}"
    );
    let after = settle(&mut world, &character, 30);
    assert_eq!(
        after.support,
        Some(ground),
        "a placed character must name the ground it lands on"
    );
}

#[test]
fn a_ceiling_stops_a_rising_platform_from_carrying_the_character_through_it() {
    let mut world = new_world(gravity_config());
    platform(&mut world, [0.0, 1.0, 0.0]);
    let ceiling = world.spawn(
        BodyDesc::cuboid([6.0, 0.5, 6.0])
            .mass(0.0)
            .position([0.0, 3.0, 0.0]),
    );
    let character = Character::spawn(&mut world, [0.0, 1.2, 0.0], CharacterDesc::default());
    let landed = settle(&mut world, &character, 20);
    assert!(landed.grounded, "the character must land on the platform");
    let underside = world.read_state(ceiling).position[1] - 0.5;
    let mut highest = landed.position[1];
    for _ in 0..150 {
        character.drive(&mut world, [0.0; 3], false);
        world.step(DT);
        world.wait();
        highest = highest.max(character.inspect_state(&mut world).position[1]);
    }
    assert!(
        highest + 0.9 <= underside + 0.03,
        "a ceiling must stop the character it reaches, the character reached {highest} against {underside}"
    );
}
