use super::common::{DT, sim, static_config};
use dynamis_model::BodyDesc;

#[test]
fn batch_spawn_lands_every_row() {
    let mut world = sim(1024, static_config());
    let mut handles = Vec::new();
    for index in 0..512 {
        let x = (index % 32) as f32 * 0.5;
        let z = (index / 32) as f32 * 0.5;
        handles.push(world.spawn(BodyDesc::sphere(0.2).position([x, 1.0 + z, z])));
    }
    world.step(DT);
    world.wait();
    for (index, handle) in handles.iter().enumerate() {
        let state = world.read_state(*handle);
        let x = (index % 32) as f32 * 0.5;
        let z = (index / 32) as f32 * 0.5;
        assert_eq!(state.position[0], x, "row {index} landed at the wrong x");
        assert_eq!(state.position[2], z, "row {index} landed at the wrong z");
        assert!(
            (state.position[1] - (1.0 + z)).abs() < 1e-6,
            "row {index} landed at the wrong y"
        );
    }
}

#[test]
fn structural_shuffle_keeps_identity() {
    let mut world = sim(16, static_config());
    let first = world.spawn(BodyDesc::sphere(0.2).position([1.0, 0.0, 0.0]));
    let second = world.spawn(BodyDesc::sphere(0.2).position([2.0, 0.0, 0.0]));
    let third = world.spawn(BodyDesc::sphere(0.2).position([3.0, 0.0, 0.0]));
    let fourth = world.spawn(BodyDesc::sphere(0.2).position([4.0, 0.0, 0.0]));

    world.remove(second);
    world.step(DT);
    world.wait();
    assert!((world.read_state(first).position[0] - 1.0).abs() < 1e-6);
    assert!((world.read_state(third).position[0] - 3.0).abs() < 1e-6);
    assert!((world.read_state(fourth).position[0] - 4.0).abs() < 1e-6);
    assert_eq!(world.count(), 3);
    assert!(world.bodies().contains(&first));
    assert!(!world.bodies().contains(&second));
}

#[test]
fn ordered_commands_on_one_slot_fold_in_order() {
    let mut world = sim(8, static_config());
    let body = world.spawn(BodyDesc::sphere(0.5).mass(2.0).position([0.0, 0.0, 0.0]));

    world.set_velocity(body, [1.0, 0.0, 0.0]);
    world.apply_force(body, [4.0, 0.0, 0.0]);
    world.apply_impulse(body, [2.0, 0.0, 0.0]);
    world.set_velocity(body, [3.0, 0.0, 0.0]);
    world.step(DT);
    world.wait();
    let state = world.read_state(body);
    let expected = 3.0 + 4.0 / 2.0 * DT;
    assert!(
        (state.velocity[0] - expected).abs() < 1e-5,
        "ordered commands must fold in order, got {}",
        state.velocity[0]
    );
    assert!(state.position[0].abs() < 1e-6 || state.position[0] > 0.0);
}

#[test]
fn edits_after_shuffle_land_on_the_moved_row() {
    let mut world = sim(8, static_config());
    let first = world.spawn(BodyDesc::sphere(0.5).position([0.0, 0.0, 0.0]));
    let second = world.spawn(BodyDesc::sphere(0.5).position([5.0, 0.0, 0.0]));

    world.remove(first);
    world.set_velocity(second, [7.0, 0.0, 0.0]);
    world.step(DT);
    world.wait();
    let state = world.read_state(second);
    assert!((state.velocity[0] - 7.0).abs() < 1e-6);
    assert!(
        (state.position[0] - (5.0 + 7.0 * DT)).abs() < 1e-5,
        "the moved row must keep its own position, got {}",
        state.position[0]
    );
}

#[test]
fn edits_before_a_shuffle_follow_their_row() {
    let mut world = sim(8, static_config());
    let first = world.spawn(BodyDesc::sphere(0.5).position([1.0, 0.0, 0.0]));
    let second = world.spawn(BodyDesc::sphere(0.5).position([2.0, 0.0, 0.0]));
    let third = world.spawn(BodyDesc::sphere(0.5).position([3.0, 0.0, 0.0]));

    world.set_velocity(third, [9.0, 0.0, 0.0]);
    world.remove(first);
    world.step(DT);
    world.wait();
    let moved = world.read_state(third);
    assert!(
        (moved.velocity[0] - 9.0).abs() < 1e-6,
        "the edit must follow the moved row, got {}",
        moved.velocity[0]
    );
    assert!((moved.position[0] - (3.0 + 9.0 * DT)).abs() < 1e-5);

    let stayed = world.read_state(second);
    assert!((stayed.velocity[0]).abs() < 1e-6);
    assert!((stayed.position[0] - 2.0).abs() < 1e-6);
}

#[test]
fn edits_on_a_removed_row_are_dropped() {
    let mut world = sim(8, static_config());
    let doomed = world.spawn(BodyDesc::sphere(0.5).position([1.0, 0.0, 0.0]));
    let survivor = world.spawn(BodyDesc::sphere(0.5).position([2.0, 0.0, 0.0]));

    world.set_velocity(doomed, [5.0, 0.0, 0.0]);
    world.remove(doomed);
    world.step(DT);
    world.wait();
    let state = world.read_state(survivor);
    assert!(
        state.velocity[0].abs() < 1e-6,
        "a removed row's edit must not leak, got {}",
        state.velocity[0]
    );
    assert!((state.position[0] - 2.0).abs() < 1e-6);
}
