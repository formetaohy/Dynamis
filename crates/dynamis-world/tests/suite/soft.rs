use super::common::{DT, distance, gravity_config, new_world, settle};
use dynamis_model::{BodyDesc, SoftBodyDesc, SoftMaterial};

fn chain() -> SoftBodyDesc {
    SoftBodyDesc::net(
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
        SoftBodyDesc::net(
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
        SoftBodyDesc::lattice([3, 3, 3], 0.25, SoftMaterial::rigid())
            .radius(0.12)
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
        let handle = world.add_soft_body(
            SoftBodyDesc::lattice([2, 2, 2], 0.4, SoftMaterial::rigid())
                .radius(0.1)
                .position([0.0, 1.5, 0.0]),
        );
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
    world.add_soft_body(SoftBodyDesc::lattice([5, 5, 5], 0.3, SoftMaterial::rigid()).radius(0.1));
    world.step(DT);
    world.wait();
    let planned = world.stream_capacity().soft;
    assert!(
        planned.particles > floor.particles,
        "the particle stream must widen for a larger soft body, {floor:?} -> {planned:?}"
    );
    assert!(
        planned.elements > floor.elements,
        "the element stream must widen for a larger soft body, {floor:?} -> {planned:?}"
    );
}

#[test]
fn two_identical_soft_bodies_observe_identical_positions() {
    let mut first = new_world(gravity_config());
    let mut second = new_world(gravity_config());
    let desc = SoftBodyDesc::lattice([3, 3, 3], 0.25, SoftMaterial::rigid())
        .radius(0.12)
        .position([0.0, 3.0, 0.0]);
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

#[test]
fn overlapping_particles_of_a_body_push_apart() {
    let mut world = new_world(super::common::static_config());
    let handle = world.add_soft_body(
        SoftBodyDesc::new(vec![[-0.3, 0.0, 0.0], [0.3, 0.0, 0.0]], Vec::new()).radius(0.5),
    );
    settle(&mut world, 60);
    let after = world.soft_body_positions(handle);
    let separation = distance(after[0], after[1]);
    assert!(
        separation > 0.9,
        "unlinked particles of one body must self collide, got {separation}"
    );
}

#[test]
fn overlapping_soft_bodies_push_apart() {
    let mut world = new_world(super::common::static_config());
    let first =
        world.add_soft_body(SoftBodyDesc::new(vec![[-0.3, 0.0, 0.0]], Vec::new()).radius(0.5));
    let second =
        world.add_soft_body(SoftBodyDesc::new(vec![[0.3, 0.0, 0.0]], Vec::new()).radius(0.5));
    settle(&mut world, 60);
    let separation = distance(
        world.soft_body_positions(first)[0],
        world.soft_body_positions(second)[0],
    );
    assert!(
        separation > 0.9,
        "soft bodies must collide with each other, got {separation}"
    );
}

#[test]
fn a_cloth_holds_its_area_while_a_distance_net_sags_flat() {
    let mut sags = Vec::new();
    for shear in [false, true] {
        let mut world = new_world(gravity_config());
        let mut desc = if shear {
            SoftBodyDesc::cloth([5, 5], 1.0, SoftMaterial::rigid())
                .radius(0.05)
                .position([0.0, 4.0, 0.0])
        } else {
            let mut links = Vec::new();
            for row in 0..5u32 {
                for column in 0..5u32 {
                    let index = column * 5 + row;
                    if row + 1 < 5 {
                        links.push([index, index + 1]);
                    }
                    if column + 1 < 5 {
                        links.push([index, index + 5]);
                    }
                }
            }
            let particles = (0..25)
                .map(|index| [(index % 5) as f32, 0.0, (index / 5) as f32])
                .collect::<Vec<_>>();
            SoftBodyDesc::net(particles, links)
                .radius(0.05)
                .position([0.0, 4.0, 0.0])
        };
        desc = desc.pinned(&[0, 4, 20, 24]);
        let handle = world.add_soft_body(desc);
        settle(&mut world, 150);
        let positions = world.soft_body_positions(handle);
        let corners = [
            positions[0][1],
            positions[4][1],
            positions[20][1],
            positions[24][1],
        ];
        let anchor = corners.iter().copied().fold(f32::MIN, f32::max);
        sags.push(anchor - positions[12][1]);
        println!("probe shear={shear} center sag {:.4}", sags.last().unwrap());
    }
    assert!(
        sags[0] > 0.6,
        "a distance net must shear and sag between its pinned corners, got {:.4}",
        sags[0]
    );
    assert!(
        sags[1] < 0.6,
        "a cloth must hold its shape between its pinned corners, got {:.4}",
        sags[1]
    );
}

#[test]
fn a_compliant_distance_element_stretches_by_its_hookean_elongation() {
    let compliance = 0.01;
    let mut world = new_world(gravity_config());
    let handle = world.add_soft_body(
        SoftBodyDesc::new(
            vec![[0.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
            vec![dynamis_model::SoftElement::distance(0, 1, 1.0).compliance(compliance)],
        )
        .pinned(&[0]),
    );
    settle(&mut world, 400);
    let positions = world.soft_body_positions(handle);
    let elongation = distance(positions[0], positions[1]) - 1.0;
    let expected = 9.81 * compliance;
    assert!(
        (elongation - expected).abs() < 0.02,
        "a compliant element must settle at its Hookean elongation {expected}, got {elongation}"
    );
}

#[test]
fn the_compliant_equilibrium_is_independent_of_the_iteration_count() {
    let compliance = 0.01;
    let mut elongations = Vec::new();
    for soft_iterations in [2u32, 8] {
        let mut world = new_world(dynamis_model::PhysicsConfig {
            soft_iterations,
            ..gravity_config()
        });
        let handle = world.add_soft_body(
            SoftBodyDesc::new(
                vec![[0.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
                vec![dynamis_model::SoftElement::distance(0, 1, 1.0).compliance(compliance)],
            )
            .pinned(&[0]),
        );
        settle(&mut world, 400);
        let positions = world.soft_body_positions(handle);
        elongations.push(distance(positions[0], positions[1]) - 1.0);
    }
    assert!(
        (elongations[0] - elongations[1]).abs() < 0.005,
        "the compliant equilibrium must not depend on the iteration count, got {elongations:?}"
    );
}
#[test]
fn a_tetrahedral_lattice_resists_compression_while_a_net_collapses() {
    let mut heights = Vec::new();
    for braced in [false, true] {
        let mut world = new_world(gravity_config());
        world.spawn(
            BodyDesc::cuboid([20.0, 0.5, 20.0])
                .mass(0.0)
                .position([0.0, -0.5, 0.0]),
        );
        let desc = if braced {
            SoftBodyDesc::lattice([3, 3, 3], 0.4, SoftMaterial::rigid())
                .radius(0.1)
                .position([0.0, 2.0, 0.0])
        } else {
            let mut links = Vec::new();
            let index = |row: u32, column: u32, layer: u32| (layer * 3 + column) * 3 + row;
            for layer in 0..3u32 {
                for column in 0..3u32 {
                    for row in 0..3u32 {
                        if row + 1 < 3 {
                            links.push([index(row, column, layer), index(row + 1, column, layer)]);
                        }
                        if column + 1 < 3 {
                            links.push([index(row, column, layer), index(row, column + 1, layer)]);
                        }
                        if layer + 1 < 3 {
                            links.push([index(row, column, layer), index(row, column, layer + 1)]);
                        }
                    }
                }
            }
            let particles = (0..27)
                .map(|slot| {
                    [
                        (slot % 3) as f32 * 0.4,
                        (slot / 9) as f32 * 0.4,
                        (slot / 3 % 3) as f32 * 0.4,
                    ]
                })
                .collect::<Vec<_>>();
            SoftBodyDesc::net(particles, links)
                .radius(0.1)
                .position([0.0, 2.0, 0.0])
        };
        let pinned = [0, 1, 2, 3, 4, 5, 6, 7, 8];
        let handle = world.add_soft_body(desc.pinned(&pinned));
        settle(&mut world, 200);
        let positions = world.soft_body_positions(handle);
        let lowest = positions
            .iter()
            .map(|position| position[1])
            .fold(f32::MAX, f32::min);
        let highest = positions
            .iter()
            .map(|position| position[1])
            .fold(f32::MIN, f32::max);
        heights.push(highest - lowest);
    }
    assert!(
        heights[1] > heights[0] * 1.2,
        "shear bracing must keep the lattice taller than an axis-only net, {heights:?}"
    );
}
