use super::common::{DT, distance, gravity_config, new_world, settle};
use dynamis_model::{BodyDesc, SoftBodyDesc};

fn chain() -> SoftBodyDesc {
    SoftBodyDesc::new(
        vec![
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0],
        ],
        vec![[0, 1], [1, 2], [2, 3]],
    )
    .radius(0.1)
    .position([0.0, 5.0, 0.0])
}

fn link_lengths(positions: &[[f32; 3]]) -> [f32; 3] {
    [
        distance(positions[0], positions[1]),
        distance(positions[1], positions[2]),
        distance(positions[2], positions[3]),
    ]
}

#[test]
fn a_free_soft_body_keeps_its_link_lengths_while_it_falls() {
    let mut world = new_world(gravity_config());
    let handle = world.add_soft_body(chain());
    let before = world.soft_body_positions(handle);
    assert_eq!(before.len(), 4, "every particle must be observed");
    settle(&mut world, 60);
    let after = world.soft_body_positions(handle);
    assert!(
        after[0][1] < before[0][1] - 1.0,
        "a free soft body must fall, got {:?}",
        after[0]
    );
    for (index, length) in link_lengths(&after).iter().enumerate() {
        assert!(
            (length - 1.0).abs() < 0.05,
            "link {index} must hold its rest length, got {length}"
        );
    }
}

#[test]
fn a_soft_body_rests_on_static_ground() {
    let mut world = new_world(gravity_config());
    world.spawn(
        BodyDesc::cuboid([20.0, 0.5, 20.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let handle = world.add_soft_body(
        SoftBodyDesc::new(
            vec![
                [-0.25, 0.0, -0.25],
                [0.25, 0.0, -0.25],
                [0.25, 0.0, 0.25],
                [-0.25, 0.0, 0.25],
                [-0.25, 0.5, -0.25],
                [0.25, 0.5, -0.25],
                [0.25, 0.5, 0.25],
                [-0.25, 0.5, 0.25],
            ],
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
            ],
        )
        .radius(0.15)
        .position([0.0, 1.5, 0.0]),
    );
    settle(&mut world, 180);
    let positions = world.soft_body_positions(handle);
    for (index, position) in positions.iter().enumerate() {
        assert!(
            position[1] >= -0.01,
            "particle {index} must stay above the ground, got {position:?}"
        );
    }
    assert!(
        positions.iter().all(|position| position[1] < 1.5),
        "the soft body must fall onto the ground"
    );
}

#[test]
fn a_soft_body_pushes_a_dynamic_body() {
    let mut world = new_world(gravity_config());
    let box_body = world.spawn(
        BodyDesc::cuboid([0.4, 0.4, 0.4])
            .position([0.0, 0.0, 0.0])
            .gravity_scale(0.0),
    );
    let handle = world.add_soft_body(
        SoftBodyDesc::lattice([3, 3, 3], 0.25, 0.12)
            .position([-1.5, 0.0, 0.0])
            .velocity([4.0, 0.0, 0.0]),
    );
    settle(&mut world, 90);
    let positions = world.soft_body_positions(handle);
    let after = world.read_state(box_body).position;
    assert!(
        after[0] > 0.05,
        "the box must be pushed along the impact direction, got {after:?}"
    );
    assert!(
        positions.iter().all(|position| position[0] < after[0]),
        "the soft body must stay behind the body it pushes"
    );
    assert_eq!(world.soft_body_count(), 1);
}

#[test]
fn a_removed_soft_body_leaves_the_world_steppable() {
    let mut world = new_world(gravity_config());
    world.spawn(
        BodyDesc::cuboid([20.0, 0.5, 20.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let handle = world.add_soft_body(chain());
    settle(&mut world, 30);
    world.remove_soft_body(handle);
    assert_eq!(world.soft_body_count(), 0);
    settle(&mut world, 30);
    assert_eq!(world.soft_body_count(), 0);
}

#[test]
fn a_soft_body_rests_on_mesh_and_plane_geometry() {
    for world_geometry in ["mesh", "plane"] {
        let mut world = new_world(gravity_config());
        if world_geometry == "mesh" {
            super::common::flat_mesh_floor(&mut world);
        } else {
            world.spawn(
                BodyDesc::new(dynamis_model::ColliderDesc::new(
                    dynamis_model::Shape::plane(),
                ))
                .mass(0.0),
            );
        }
        let handle = world
            .add_soft_body(SoftBodyDesc::lattice([2, 2, 2], 0.4, 0.1).position([0.0, 1.5, 0.0]));
        settle(&mut world, 180);
        let positions = world.soft_body_positions(handle);
        for (index, position) in positions.iter().enumerate() {
            assert!(
                position[1] >= -0.02,
                "{world_geometry}: particle {index} must stay above the surface, got {position:?}"
            );
        }
        let highest = positions
            .iter()
            .map(|position| position[1])
            .fold(f32::MIN, f32::max);
        assert!(
            highest < 1.5,
            "{world_geometry}: the soft body must settle onto the surface"
        );
    }
}

#[test]
fn a_wider_soft_body_widens_the_soft_streams() {
    let mut world = new_world(gravity_config());
    let floor = world.stream_capacity().soft;
    world.add_soft_body(SoftBodyDesc::lattice([5, 5, 5], 0.3, 0.1));
    world.step(DT);
    world.wait();
    let planned = world.stream_capacity().soft;
    assert!(
        planned.particles > floor.particles,
        "the particle stream must widen for a larger soft body, {floor:?} -> {planned:?}"
    );
    assert!(
        planned.links > floor.links,
        "the link stream must widen for a larger soft body, {floor:?} -> {planned:?}"
    );
}

#[test]
fn two_identical_soft_bodies_observe_identical_positions() {
    let mut first = new_world(gravity_config());
    let mut second = new_world(gravity_config());
    let desc = SoftBodyDesc::lattice([3, 3, 3], 0.25, 0.12).position([0.0, 3.0, 0.0]);
    let first_body = first.add_soft_body(desc.clone());
    let second_body = second.add_soft_body(desc);
    let first_start = first.soft_body_positions(first_body);
    let second_start = second.soft_body_positions(second_body);
    assert_eq!(first_start, second_start);
    settle(&mut first, 75);
    settle(&mut second, 75);
    assert_eq!(
        first.soft_body_positions(first_body),
        second.soft_body_positions(second_body),
        "identical soft bodies must stay bit identical"
    );
}
