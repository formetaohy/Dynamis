use super::common::{DT, new_world, static_config};
use dynamis_layout::{
    COUNTER_BODY_EDITS, COUNTER_BODY_MOVES, COUNTER_CONSTRAINT_MOVES, COUNTER_CONSTRAINTS,
};
use dynamis_model::{BodyDesc, ConstraintDesc};

#[test]
fn batch_spawn_lands_every_row() {
    let mut world = new_world(static_config());
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
    let mut world = new_world(static_config());
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
    let mut world = new_world(static_config());
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
    let mut world = new_world(static_config());
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
    let mut world = new_world(static_config());
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
    let mut world = new_world(static_config());
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

#[test]
fn an_edit_declares_one_stream_row() {
    let mut world = new_world(static_config());
    let edited = world.spawn(BodyDesc::sphere(0.5).position([1.0, 0.0, 0.0]));
    let untouched = world.spawn(BodyDesc::sphere(0.5).position([5.0, 0.0, 0.0]));

    world.set_velocity(edited, [3.0, 0.0, 0.0]);
    world.step(DT);
    world.wait();
    assert_eq!(
        world.measured()[COUNTER_BODY_EDITS],
        1,
        "one edited row must declare exactly one stream row"
    );
    assert!((world.read_state(edited).velocity[0] - 3.0).abs() < 1e-6);
    assert!(
        world.read_state(untouched).velocity[0].abs() < 1e-6,
        "an edit must never spill onto another row"
    );
}

#[test]
fn a_step_without_edits_declares_an_empty_stream() {
    let mut world = new_world(static_config());
    let body = world.spawn(BodyDesc::sphere(0.5).position([0.0, 0.0, 0.0]));
    world.step(DT);
    world.wait();
    assert_eq!(world.measured()[COUNTER_BODY_EDITS], 0);
    for _ in 0..3 {
        world.step(DT);
    }
    world.wait();
    assert_eq!(
        world.measured()[COUNTER_BODY_EDITS],
        0,
        "a quiet step must not declare an edit stream"
    );
    assert!((world.read_state(body).position[0]).abs() < 1e-6);
}

#[test]
fn forced_edits_reach_every_row_of_a_large_world() {
    let mut world = new_world(static_config());
    let bodies = (0..64)
        .map(|index| {
            world.spawn(
                BodyDesc::sphere(0.5)
                    .mass(2.0)
                    .position([index as f32 * 4.0, 0.0, 0.0]),
            )
        })
        .collect::<Vec<_>>();
    for (index, handle) in bodies.iter().enumerate() {
        world.apply_impulse(*handle, [index as f32 + 1.0, 0.0, 0.0]);
    }
    world.step(DT);
    world.wait();
    assert_eq!(
        world.measured()[COUNTER_BODY_EDITS],
        bodies.len() as u32,
        "every edited row must be present in the stream"
    );
    for (index, handle) in bodies.iter().enumerate() {
        let expected = (index as f32 + 1.0) / 2.0;
        assert!(
            (world.read_state(*handle).velocity[0] - expected).abs() < 1e-6,
            "row {index} missed its impulse"
        );
    }
}

#[test]
fn constraint_edits_declare_their_own_stream() {
    let mut world = new_world(static_config());
    let first = world.spawn(BodyDesc::sphere(0.5));
    let second = world.spawn(BodyDesc::sphere(0.5).position([0.0, 1.0, 0.0]));
    world.add_constraint(
        first,
        second,
        ConstraintDesc::distance([0.0; 3], [0.0; 3], 1.0),
    );
    world.step(DT);
    world.wait();
    assert_eq!(world.measured()[COUNTER_CONSTRAINTS], 1);
    assert_eq!(world.constraints().len(), 1);
}

#[test]
fn a_row_move_stream_declares_only_touched_rows() {
    let mut world = new_world(static_config());
    let _ground = world.spawn(BodyDesc::static_sphere(0.5));
    let first = world.spawn(BodyDesc::sphere(0.25).position([1.0, 0.0, 0.0]));
    let second = world.spawn(BodyDesc::sphere(0.25).position([2.0, 0.0, 0.0]));
    world.set_velocity(second, [3.0, 0.0, 0.0]);
    world.step(DT);
    world.wait();
    assert_eq!(
        world.measured()[COUNTER_BODY_MOVES],
        3,
        "spawning into the dynamic partition must declare only the shuffled rows"
    );
    assert_eq!(world.measured()[COUNTER_BODY_EDITS], 1);
    assert!((world.read_state(second).velocity[0] - 3.0).abs() < 1e-6);

    world.remove(first);
    world.step(DT);
    world.wait();
    assert_eq!(
        world.measured()[COUNTER_BODY_MOVES],
        2,
        "removing a row must declare the surviving tail rows only"
    );
    assert!((world.read_state(second).velocity[0] - 3.0).abs() < 1e-6);
}

#[test]
fn a_large_world_shuffles_only_the_rows_touched_by_commands() {
    let mut world = new_world(static_config());
    for index in 0..256 {
        world.spawn(BodyDesc::static_sphere(0.1).position([100.0 + index as f32, 0.0, 0.0]));
    }
    world.step(DT);
    world.wait();
    let ground = world.spawn(BodyDesc::static_sphere(0.5));
    let ball = world.spawn(BodyDesc::sphere(0.25).position([1.0, 0.0, 0.0]));
    let second = world.spawn(BodyDesc::sphere(0.25).position([2.0, 0.0, 0.0]));
    let anchor = world.spawn(BodyDesc::static_sphere(0.25).position([0.0, 3.0, 0.0]));
    let joint = world.add_constraint(
        anchor,
        ball,
        ConstraintDesc::distance([0.0; 3], [0.0; 3], 1.0),
    );
    let spare = world.add_constraint(
        anchor,
        second,
        ConstraintDesc::distance([0.0; 3], [0.0; 3], 1.0),
    );
    world.remove_constraint(joint);
    world.step(DT);
    world.wait();
    assert_eq!(
        world.measured()[COUNTER_BODY_MOVES],
        6,
        "a 260 row world must shuffle only the rows touched by commands"
    );
    assert_eq!(
        world.measured()[COUNTER_CONSTRAINT_MOVES],
        1,
        "removing a non tail constraint must move only the tail row"
    );
    assert_eq!(world.count(), 260);
    assert_eq!(world.constraints(), &[spare]);
    assert_eq!(world.read_state(ground).position, [0.0, 0.0, 0.0]);
    assert_eq!(world.read_state(anchor).position, [0.0, 3.0, 0.0]);
}

#[test]
fn a_quiet_step_declares_no_row_moves() {
    let mut world = new_world(static_config());
    world.spawn(BodyDesc::sphere(0.25));
    world.step(DT);
    world.wait();
    assert!(world.measured()[COUNTER_BODY_MOVES] > 0);
    for _ in 0..3 {
        world.step(DT);
    }
    world.wait();
    assert_eq!(
        world.measured()[COUNTER_BODY_MOVES],
        0,
        "a step without structural commands must not declare row moves"
    );
}
