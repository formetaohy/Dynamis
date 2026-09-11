use super::common::{asleep, new_world, settle_until, static_config};
use dynamis_mesh::HullDecomposeSettings;
use dynamis_model::BodyDesc;

fn l_prism() -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let h = 0.5;
    let profile = [
        [0.0, 0.0],
        [2.0, 0.0],
        [2.0, 1.0],
        [1.0, 1.0],
        [1.0, 2.0],
        [0.0, 2.0],
    ];
    let mut vertices = Vec::new();
    for [x, y] in profile {
        vertices.push([x, y, -h]);
        vertices.push([x, y, h]);
    }
    let mut triangles = Vec::new();
    for i in 0..6 {
        let j = (i + 1) % 6;
        let a0 = (2 * i) as u32;
        let a1 = (2 * j) as u32;
        let a2 = (2 * j + 1) as u32;
        let a3 = (2 * i + 1) as u32;
        triangles.push([a0, a1, a2]);
        triangles.push([a0, a2, a3]);
    }
    for i in 1..5 {
        let a = 0;
        let b = i as u32;
        let c = (i + 1) as u32;
        triangles.push([2 * a, 2 * b, 2 * c]);
        triangles.push([2 * a + 1, 2 * c + 1, 2 * b + 1]);
    }
    (vertices, triangles)
}

#[test]
fn decomposition_splits_concave_mesh_into_convex_parts() {
    let mut world = new_world(static_config());
    let (vertices, triangles) = l_prism();
    let parts = world.add_decomposed_mesh(
        &vertices,
        &triangles,
        HullDecomposeSettings {
            max_parts: 16,
            concavity: 0.05,
            depth: 8,
        },
    );
    assert!(
        (2..=16).contains(&parts.len()),
        "concave L shape must split into multiple convex parts, got {}",
        parts.len()
    );
}

#[test]
fn decomposed_body_rests_on_ground() {
    let mut world = super::common::new_world(super::common::gravity_config());
    let (vertices, triangles) = l_prism();
    let parts = world.add_decomposed_mesh(&vertices, &triangles, HullDecomposeSettings::default());
    let desc = BodyDesc::compound(&parts).position([2.0, 3.0, 2.0]);
    let body = world.spawn(desc);
    world.spawn(
        BodyDesc::cuboid([20.0, 0.5, 20.0])
            .mass(0.0)
            .position([0.0, -0.5, 0.0]),
    );
    settle_until(&mut world, 600, |world| asleep(world));
    let state = world.read_state(body);
    assert!(
        state.position[1] > -0.5 && state.position[1] < 2.5,
        "decomposed body must rest on the floor, got y={}",
        state.position[1]
    );
    assert!(
        state.sleeping,
        "settled decomposed body must fall asleep on the floor, got y={} speed {}",
        state.position[1], state.velocity[1]
    );
}

#[test]
fn shape_source_refcount_blocks_removal_while_live() {
    let mut world = super::common::new_world(static_config());
    let (vertices, triangles) = l_prism();
    let parts = world.add_decomposed_mesh(&vertices, &triangles, HullDecomposeSettings::default());
    let body = world.spawn(BodyDesc::compound(&parts));
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            world.remove_shape(parts[0]);
        }))
        .is_err(),
        "removing an in-use shape must panic"
    );
    world.remove(body);
    world.remove_shape(parts[0]);
}
