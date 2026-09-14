use super::common::{DT, distance, gravity_config, new_world, settle};
use dynamis_model::{BodyDesc, SoftBodyDesc, SoftElement, SoftMaterial};

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
    let before = world.inspect_soft_particles(handle);
    assert_eq!(before.len(), 4, "every particle must be observed");
    settle(&mut world, 60);
    let after = world.inspect_soft_particles(handle);
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
    let positions = world.inspect_soft_particles(handle);
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
    let positions = world.inspect_soft_particles(handle);
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
        let positions = world.inspect_soft_particles(handle);
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
    let desc = SoftBodyDesc::lattice([3, 3, 3], 0.3, SoftMaterial::new(0.001, 0.001, 0.0, 0.0001))
        .radius(0.15)
        .position([0.0, 2.0, 0.0]);
    let first_body = first.add_soft_body(desc.clone());
    let second_body = second.add_soft_body(desc);
    super::common::static_sphere_ground(&mut first, 1.0);
    super::common::static_sphere_ground(&mut second, 1.0);
    let first_start = first.inspect_soft_particles(first_body);
    let second_start = second.inspect_soft_particles(second_body);
    assert_eq!(first_start, second_start);
    settle(&mut first, 90);
    settle(&mut second, 90);
    assert_eq!(
        first.inspect_soft_particles(first_body),
        second.inspect_soft_particles(second_body),
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
    let after = world.inspect_soft_particles(handle);
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
        world.inspect_soft_particles(first)[0],
        world.inspect_soft_particles(second)[0],
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
        let positions = world.inspect_soft_particles(handle);
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
    let positions = world.inspect_soft_particles(handle);
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
        let positions = world.inspect_soft_particles(handle);
        elongations.push(distance(positions[0], positions[1]) - 1.0);
    }
    assert!(
        (elongations[0] - elongations[1]).abs() < 0.005,
        "the compliant equilibrium must not depend on the iteration count, got {elongations:?}"
    );
}
#[test]
fn a_tetrahedral_lattice_keeps_its_shape_while_an_axis_only_net_shears_away() {
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
        heights.push(height_of(&world.inspect_soft_particles(handle)));
    }
    assert!(
        heights[1] > heights[0] * 1.05,
        "tetrahedral bracing must keep the lattice taller than an axis-only net, {heights:?}"
    );
    assert!(
        heights[1] > 0.78,
        "a rigid lattice must hold its rest height, got {}",
        heights[1]
    );
}

#[test]
fn a_volume_element_holds_its_tetrahedron_against_gravity() {
    let mut world = new_world(gravity_config());
    let handle = world.add_soft_body(probe_tetrahedron(0.0));
    settle(&mut world, 400);
    let positions = world.inspect_soft_particles(handle);
    assert!(
        (positions[3][1] - 1.0).abs() < 0.01,
        "a rigid volume element must hold its apex, got {:?}",
        positions[3]
    );
    assert!(
        (tetrahedron_volume(&positions) - 1.0 / 6.0).abs() < 0.005,
        "a rigid volume element must hold its rest volume"
    );
}

#[test]
fn a_compliant_volume_element_reaches_its_hookean_volume_deficit() {
    let compliance = 0.0002;
    let mut world = new_world(gravity_config());
    let handle = world.add_soft_body(probe_tetrahedron(compliance));
    settle(&mut world, 200);
    let mut deficit = 0.0;
    let mut samples = 0.0;
    for _ in 0..300 {
        world.step(DT);
        deficit += 1.0 / 6.0 - tetrahedron_volume(&world.inspect_soft_particles(handle));
        samples += 1.0;
    }
    let deficit = deficit / samples;
    let expected = 6.0 * compliance * 9.81;
    assert!(
        (deficit - expected).abs() < 0.002,
        "a compliant tetrahedron must sag by its Hookean volume deficit {expected}, got {deficit}"
    );
}

#[test]
fn a_volume_compliance_softens_the_lattice_under_load() {
    let mut heights = Vec::new();
    for volume in [0.0, 0.001] {
        let mut world = new_world(gravity_config());
        let handle = world.add_soft_body(
            SoftBodyDesc::lattice([3, 3, 3], 0.4, SoftMaterial::new(0.002, 0.002, 0.0, volume))
                .radius(0.0)
                .position([0.0, 2.0, 0.0])
                .pinned(&[0, 1, 2, 3, 4, 5, 6, 7, 8]),
        );
        settle(&mut world, 240);
        heights.push(height_of(&world.inspect_soft_particles(handle)));
    }
    assert!(
        heights[0] < heights[1] * 0.95,
        "a rigid volume element must hold the stretched lattice back, {heights:?}"
    );
}

#[test]
fn an_area_element_holds_its_triangle_against_gravity() {
    let mut world = new_world(gravity_config());
    let handle = world.add_soft_body(probe_triangle(0.0));
    settle(&mut world, 400);
    let positions = world.inspect_soft_particles(handle);
    assert!(
        (positions[2][1] - 1.0).abs() < 0.01,
        "a rigid area element must hold its apex, got {:?}",
        positions[2]
    );
    assert!(
        (triangle_area(&positions) - 0.5).abs() < 0.005,
        "a rigid area element must hold its rest area"
    );
}

#[test]
fn a_compliant_area_element_reaches_its_hookean_area_deficit() {
    let compliance = 0.005;
    let mut world = new_world(gravity_config());
    let handle = world.add_soft_body(probe_triangle(compliance));
    settle(&mut world, 200);
    let mut deficit = 0.0;
    let mut samples = 0.0;
    for _ in 0..300 {
        world.step(DT);
        deficit += 0.5 - triangle_area(&world.inspect_soft_particles(handle));
        samples += 1.0;
    }
    let deficit = deficit / samples;
    let expected = 2.0 * compliance * 9.81;
    assert!(
        (deficit - expected).abs() < 0.01,
        "a compliant triangle must sag by its Hookean area deficit {expected}, got {deficit}"
    );
}

#[test]
fn a_bend_element_holds_its_dihedral_angle_against_gravity() {
    let mut folds = Vec::new();
    for compliance in [0.0, 0.05] {
        let mut world = new_world(gravity_config());
        let handle = world.add_soft_body(probe_hinge(compliance));
        settle(&mut world, 200);
        let mut apex = 0.0;
        let mut samples = 0.0;
        for _ in 0..200 {
            world.step(DT);
            apex += world.inspect_soft_particles(handle)[0][1];
            samples += 1.0;
        }
        folds.push(apex / samples);
    }
    assert!(
        folds[0].abs() < 0.01,
        "a rigid bend element must hold the flap in its rest plane, got {}",
        folds[0]
    );
    assert!(
        folds[1] < -0.05,
        "a compliant bend element must let the flap fold, got {}",
        folds[1]
    );
}

#[test]
fn a_fast_soft_body_never_sinks_into_a_solid_collider() {
    let mut world = new_world(gravity_config());
    world.spawn(
        BodyDesc::cuboid([5.0, 0.5, 5.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    let handle = world.add_soft_body(
        SoftBodyDesc::lattice([3, 3, 3], 0.3, SoftMaterial::rigid())
            .radius(0.1)
            .position([0.0, 3.0, 0.0]),
    );
    let mut lowest = f32::MAX;
    for _ in 0..300 {
        world.step(DT);
        let positions = world.inspect_soft_particles(handle);
        lowest = lowest.min(
            positions
                .iter()
                .map(|position| position[1])
                .fold(f32::MAX, f32::min),
        );
    }
    assert!(
        lowest > 0.0,
        "a dropping soft body must never sink into a solid collider, got {lowest}"
    );
    assert!(
        lowest < 0.15,
        "the soft body must come to rest on the collider surface, got {lowest}"
    );
}

fn height_of(positions: &[[f32; 3]]) -> f32 {
    let lowest = positions
        .iter()
        .map(|position| position[1])
        .fold(f32::MAX, f32::min);
    let highest = positions
        .iter()
        .map(|position| position[1])
        .fold(f32::MIN, f32::max);
    highest - lowest
}

fn tetrahedron_volume(positions: &[[f32; 3]]) -> f32 {
    let first = sub(positions[1], positions[0]);
    let second = sub(positions[2], positions[0]);
    let third = sub(positions[3], positions[0]);
    dot(cross(first, second), third).abs() / 6.0
}

fn triangle_area(positions: &[[f32; 3]]) -> f32 {
    let first = sub(positions[1], positions[0]);
    let second = sub(positions[2], positions[0]);
    0.5 * length(cross(first, second))
}

fn sub(first: [f32; 3], second: [f32; 3]) -> [f32; 3] {
    [
        first[0] - second[0],
        first[1] - second[1],
        first[2] - second[2],
    ]
}

fn cross(first: [f32; 3], second: [f32; 3]) -> [f32; 3] {
    [
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    ]
}

fn dot(first: [f32; 3], second: [f32; 3]) -> f32 {
    first[0] * second[0] + first[1] * second[1] + first[2] * second[2]
}

fn length(vector: [f32; 3]) -> f32 {
    dot(vector, vector).sqrt()
}

fn probe_tetrahedron(compliance: f32) -> SoftBodyDesc {
    SoftBodyDesc::new(
        vec![
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.25, 1.0, 0.25],
        ],
        vec![SoftElement::volume(0, 1, 2, 3, 1.0 / 6.0).compliance(compliance)],
    )
    .pinned(&[0, 1, 2])
}

fn probe_triangle(compliance: f32) -> SoftBodyDesc {
    SoftBodyDesc::new(
        vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.25, 1.0, 0.0]],
        vec![SoftElement::area(0, 1, 2, 0.5).compliance(compliance)],
    )
    .pinned(&[0, 1])
}

fn probe_hinge(compliance: f32) -> SoftBodyDesc {
    SoftBodyDesc::new(
        vec![
            [0.5, 0.0, 1.0],
            [0.5, 0.0, -1.0],
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        vec![SoftElement::bend(0, 1, 2, 3, 0.0).compliance(compliance)],
    )
    .pinned(&[1, 2, 3])
}

fn hanging_link(element: SoftElement) -> SoftBodyDesc {
    SoftBodyDesc::new(vec![[0.0, 0.0, 0.0], [0.0, -1.0, 0.0]], vec![element]).pinned(&[0])
}

#[test]
fn a_yielding_link_keeps_the_stretch_it_creeps_into() {
    let compliance = 0.05;
    let mut world = new_world(dynamis_model::PhysicsConfig {
        damping: 2.0,
        ..gravity_config()
    });
    let plastic = world.add_soft_body(hanging_link(
        SoftElement::distance(0, 1, 1.0)
            .compliance(compliance)
            .yielding(0.2, 0.1),
    ));
    let elastic = world.add_soft_body(hanging_link(
        SoftElement::distance(0, 1, 1.0).compliance(compliance),
    ));
    let link_length = |world: &mut dynamis_world::World, handle| {
        let positions = world.inspect_soft_particles(handle);
        distance(positions[0], positions[1])
    };
    settle(&mut world, 400);
    let crept = world.inspect_soft_elements(plastic)[0].rest();
    let stretched = link_length(&mut world, plastic);
    println!(
        "probe rest {crept:.4} length {stretched:.4} elastic {:.4}",
        link_length(&mut world, elastic)
    );
    assert!(
        crept > 1.5,
        "a link loaded past its yield strain must creep into a new rest length, got {crept}"
    );
    let strain = stretched / crept - 1.0;
    assert!(
        strain < 0.21,
        "plastic flow must stop at the yield strain, got {strain}"
    );
    world.set_gravity([0.0, 0.0, 0.0]);
    settle(&mut world, 400);
    let kept = link_length(&mut world, plastic);
    let released = link_length(&mut world, elastic);
    println!("probe kept {kept:.4} released {released:.4} rest {crept:.4}");
    assert!(
        kept > 1.5 && (kept / crept - 1.0).abs() < 0.1,
        "a yielded link must keep its new rest length, got {kept} for {crept}"
    );
    assert!(
        (released - 1.0).abs() < 0.1,
        "an elastic link must return to its rest length, got {released}"
    );
}

#[test]
fn a_link_loaded_past_its_break_strain_tears() {
    let mut world = new_world(dynamis_model::PhysicsConfig {
        damping: 2.0,
        ..gravity_config()
    });
    let torn = world.add_soft_body(hanging_link(
        SoftElement::distance(0, 1, 1.0)
            .compliance(0.02)
            .fracturing(0.05),
    ));
    let held = world.add_soft_body(hanging_link(
        SoftElement::distance(0, 1, 1.0).compliance(0.02),
    ));
    settle(&mut world, 200);
    let elements = world.inspect_soft_elements(torn);
    assert_eq!(elements.len(), 1, "a link observes its own element");
    assert!(
        elements[0].broken(),
        "a link stretched past its break strain must fail"
    );
    let fallen = world.inspect_soft_particles(torn)[1][1];
    assert!(
        fallen < -1.6,
        "a torn particle must fall away from its anchor, got {fallen}"
    );
    assert!(!world.inspect_soft_elements(held)[0].broken());
    let suspended = world.inspect_soft_particles(held)[1][1];
    assert!(
        (suspended + 1.196).abs() < 0.02,
        "an unbreakable link must keep hanging at its Hookean elongation, got {suspended}"
    );
}

#[test]
fn a_lattice_fractures_where_its_material_fails() {
    let mut world = new_world(gravity_config());
    let material = SoftMaterial::new(0.01, 0.01, 0.0, 0.001)
        .yielding(0.05, 0.5)
        .fracturing(0.1);
    let handle = world.add_soft_body(
        SoftBodyDesc::lattice([2, 2, 2], 0.5, material)
            .radius(0.1)
            .position([0.0, 3.0, 0.0])
            .pinned(&[0]),
    );
    settle(&mut world, 120);
    let elements = world.inspect_soft_elements(handle);
    assert!(
        elements.iter().any(|element| element.broken()),
        "a lattice pinned below its load must tear"
    );
    let positions = world.inspect_soft_particles(handle);
    assert!(
        positions[7][1] < 1.0,
        "a fractured lattice must fall away from its pin, got {:?}",
        positions[7]
    );
}

#[test]
fn a_yielding_body_stays_bit_identical_across_worlds() {
    let mut first = new_world(gravity_config());
    let mut second = new_world(gravity_config());
    let desc = hanging_link(
        SoftElement::distance(0, 1, 1.0)
            .compliance(0.05)
            .yielding(0.2, 0.25),
    );
    let first_body = first.add_soft_body(desc.clone());
    let second_body = second.add_soft_body(desc);
    settle(&mut first, 240);
    settle(&mut second, 240);
    let rests = |state: &dynamis_model::SoftElementState| state.rest().to_bits();
    assert_eq!(
        first
            .inspect_soft_elements(first_body)
            .iter()
            .map(rests)
            .collect::<Vec<_>>(),
        second
            .inspect_soft_elements(second_body)
            .iter()
            .map(rests)
            .collect::<Vec<_>>(),
        "plastic flow must stay bit identical"
    );
    assert_eq!(
        first.inspect_soft_particles(first_body),
        second.inspect_soft_particles(second_body)
    );
}

#[test]
fn two_element_bodies_keep_their_own_particles() {
    let mut world = new_world(gravity_config());
    let bodies = [
        world.add_soft_body(
            SoftBodyDesc::lattice([2, 2, 2], 0.5, SoftMaterial::rigid())
                .radius(0.1)
                .position([-3.0, 1.0, 0.0]),
        ),
        world.add_soft_body(
            SoftBodyDesc::lattice([2, 2, 2], 0.5, SoftMaterial::rigid())
                .radius(0.1)
                .position([3.0, 1.0, 0.0]),
        ),
    ];
    let before = bodies.map(|handle| world.inspect_soft_particles(handle));
    settle(&mut world, 60);
    for (body, original) in bodies.into_iter().zip(before) {
        let after = world.inspect_soft_particles(body);
        let mut drift = 0.0f32;
        for first in 0..after.len() {
            for second in first + 1..after.len() {
                let held = distance(original[first], original[second]);
                let moved = distance(after[first], after[second]);
                drift = drift.max((held - moved).abs());
            }
        }
        assert!(
            drift < 0.05,
            "a rigid lattice must keep its own span while it falls, drifted {drift}"
        );
    }
}

#[test]
fn a_snapshot_keeps_the_material_state_a_body_crept_into() {
    let mut world = new_world(dynamis_model::PhysicsConfig {
        damping: 2.0,
        ..gravity_config()
    });
    let handle = world.add_soft_body(hanging_link(
        SoftElement::distance(0, 1, 1.0)
            .compliance(0.02)
            .fracturing(0.05),
    ));
    let snapshot = world.snapshot();
    settle(&mut world, 60);
    assert!(
        world.inspect_soft_elements(handle)[0].broken(),
        "a link loaded past its break strain must fail"
    );
    world.restore(&snapshot);
    assert!(
        !world.inspect_soft_elements(handle)[0].broken(),
        "a restored world must keep the link its snapshot captured intact"
    );
    settle(&mut world, 60);
    assert!(
        world.inspect_soft_elements(handle)[0].broken(),
        "a restored world must fail the same link again"
    );
}

fn lattice(side: u32) -> (Vec<[f32; 3]>, Vec<[u32; 2]>) {
    let mut vertices = Vec::new();
    for row in 0..side {
        for col in 0..side {
            vertices.push([col as f32 * 0.2, 0.0, row as f32 * 0.2]);
        }
    }
    let mut links = Vec::new();
    let at = |row: u32, col: u32| row * side + col;
    for row in 0..side {
        for col in 0..side {
            if col + 1 < side {
                links.push([at(row, col), at(row, col + 1)]);
            }
            if row + 1 < side {
                links.push([at(row, col), at(row + 1, col)]);
            }
            if col + 1 < side && row + 1 < side {
                links.push([at(row, col), at(row + 1, col + 1)]);
            }
        }
    }
    (vertices, links)
}

#[test]
fn a_fresh_soft_body_replaces_the_run_a_removed_one_retires() {
    let mut world = new_world(gravity_config());
    let (vertices, links) = lattice(24);
    let retired =
        world.add_soft_body(SoftBodyDesc::net(vertices, links).position([0.0, 20.0, 0.0]));
    settle(&mut world, 3);
    world.remove_soft_body(retired);
    let fresh = world.add_soft_body(chain());
    settle(&mut world, 30);
    assert_eq!(
        world.soft_body_count(),
        1,
        "only the fresh body must remain"
    );
    let positions = world.inspect_soft_particles(fresh);
    assert_eq!(positions.len(), 4, "the fresh body must own every particle");
    assert!(
        positions[0][1] < 4.0,
        "the fresh body must simulate in the run the removed one retired, got {:?}",
        positions[0]
    );
    for (index, length) in link_lengths(&positions).iter().enumerate() {
        assert!(
            (length - 1.0).abs() < 0.05,
            "link {index} must hold its rest length, got {length}"
        );
    }
}
