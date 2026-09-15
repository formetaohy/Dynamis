use super::common::{DT, new_world, static_config};
use dynamis_model::{
    BodyDesc, BodyHandle, CharacterDesc, CharacterHandle, CharacterInput, ConstraintDesc,
    VehicleDesc, VehicleHandle, WheelDesc,
};
use dynamis_world::World;
use std::panic::{AssertUnwindSafe, catch_unwind};

fn refuses_removal(world: &mut World, body: BodyHandle, why: &str) {
    assert!(
        catch_unwind(AssertUnwindSafe(|| world.remove(body))).is_err(),
        "{why} must refuse the removal of its body"
    );
    assert!(
        world.bodies().contains(&body),
        "{why} must leave the refused body alive"
    );
}

fn character(world: &mut World) -> CharacterHandle {
    world.add_character([0.0, 1.0, 0.0], CharacterDesc::default())
}

fn vehicle(world: &mut World) -> VehicleHandle {
    world.add_vehicle(VehicleDesc::new(
        BodyDesc::cuboid([0.9, 0.3, 1.8])
            .mass(900.0)
            .position([0.0, 0.75, 0.0]),
        [WheelDesc::new([0.8, -0.2, 1.2], 0.35).driving()].to_vec(),
    ))
}

fn inherited(body: BodyHandle, world: &mut World) -> BodyHandle {
    let recycled = world.spawn(BodyDesc::sphere(0.5).position([40.0, 3.0, 40.0]));
    assert_eq!(
        recycled.id, body.id,
        "the retirement must free the body id for reuse"
    );
    for _ in 0..8 {
        world.step(DT);
        world.wait();
    }
    let state = world.read_state(recycled);
    assert_eq!(
        (state.position[0], state.position[2]),
        (40.0, 40.0),
        "a retired owner must not drive the body that inherited its id, got {:?}",
        state.position
    );
    recycled
}

#[test]
fn a_body_a_live_constraint_holds_refuses_removal() {
    let mut world = new_world(static_config());
    let first = world.spawn(BodyDesc::sphere(0.4));
    let second = world.spawn(BodyDesc::sphere(0.4).position([0.0, 1.0, 0.0]));
    world.add_constraint(
        first,
        second,
        ConstraintDesc::distance([0.0; 3], [0.0; 3], 1.0),
    );
    refuses_removal(&mut world, first, "a body a live constraint holds");
    refuses_removal(&mut world, second, "a body a live constraint holds");
    world.remove_constraint(world.body_constraints(first)[0]);
    world.remove(first);
}

#[test]
fn a_body_a_live_character_drives_refuses_removal() {
    let mut world = new_world(static_config());
    let character = character(&mut world);
    let body = world.character_body(character);
    refuses_removal(&mut world, body, "a body a live character drives");
    world.remove_character(character);
}

#[test]
fn a_retired_character_hands_its_recycled_body_to_its_next_owner() {
    let mut world = new_world(static_config());
    let character = character(&mut world);
    let body = world.character_body(character);
    for _ in 0..8 {
        world.set_character_input(character, CharacterInput::moving([1.0, 0.0, 0.0]));
        world.step(DT);
    }
    world.wait();
    assert!(
        world.read_state(body).position[0] > 0.1,
        "the character must drive the body it owns"
    );

    world.remove_character(character);
    assert!(
        catch_unwind(AssertUnwindSafe(|| world.character_body(character))).is_err(),
        "a retired character must refuse its own handle"
    );
    inherited(body, &mut world);
}

#[test]
fn a_retired_vehicle_hands_its_recycled_body_to_its_next_owner() {
    let mut world = new_world(static_config());
    let vehicle = vehicle(&mut world);
    let chassis = world.vehicle_body(vehicle);
    for _ in 0..4 {
        world.step(DT);
    }
    world.wait();

    world.remove_vehicle(vehicle);
    assert!(
        catch_unwind(AssertUnwindSafe(|| world.vehicle_body(vehicle))).is_err(),
        "a retired vehicle must refuse its own handle"
    );
    inherited(chassis, &mut world);
}

#[test]
fn a_released_body_owner_keeps_its_own_observation_alone() {
    let mut world = new_world(static_config());
    let first = character(&mut world);
    let second = character(&mut world);
    world.set_character_input(first, CharacterInput::moving([1.0, 0.0, 0.0]));
    world.set_character_input(second, CharacterInput::moving([0.0, 0.0, 1.0]));
    assert!(
        world.try_character_state(second).is_none(),
        "a character state that no run has published must not be invented by the host"
    );
    world.step(DT);
    world.wait();
    assert!(
        world.try_character_state(second).is_some(),
        "wait() must publish the state of a watched character"
    );

    world.stop_observing_character(first);
    world.step(DT);
    world.wait();
    assert!(
        world.try_character_state(second).is_some(),
        "releasing one character must keep publishing the others"
    );

    world.remove_character(second);
    assert!(
        catch_unwind(AssertUnwindSafe(|| world.try_character_state(second))).is_err(),
        "a retired character must refuse its own observation"
    );
}
