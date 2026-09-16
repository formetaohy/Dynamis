use super::common::{DT, gravity_config, observed_world, settle, static_config};
use dynamis_model::{BodyDesc, ColliderDesc, QueryFilter, Shape, SurfaceDesc, SurfaceTable};
use dynamis_world::{ContactManifold, ContactPoint, World};

const ICE: f32 = 0.04;
const ROCK: f32 = 1.2;

fn ice() -> SurfaceDesc {
    SurfaceDesc::new().friction(ICE).restitution(0.0)
}

fn rock() -> SurfaceDesc {
    SurfaceDesc::new().friction(ROCK).restitution(0.0)
}

fn palette() -> [SurfaceDesc; 2] {
    [ice(), rock()]
}

fn patch(centre: f32, height: f32) -> Vec<[f32; 3]> {
    vec![
        [centre - 3.0, height, -4.0],
        [centre + 3.0, height, -4.0],
        [centre + 3.0, height, 4.0],
    ]
}

fn two_patches() -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let mut vertices = patch(0.0, 0.0);
    vertices.extend(patch(8.0, 3.0));
    (vertices, vec![[0u32, 1, 2], [3, 4, 5]])
}

fn deepest(manifold: &ContactManifold) -> &ContactPoint {
    manifold
        .points
        .iter()
        .max_by(|left, right| left.depth.total_cmp(&right.depth))
        .expect("a manifold holds at least one point")
}

fn flat_mesh(world: &mut World, height: f32, indices: &[u32]) -> dynamis_model::ShapeSourceHandle {
    let vertices = patch(0.0, height);
    let triangles = vec![[0u32, 1, 2]];
    let palette = palette();
    let floor = world.add_mesh(
        &vertices,
        &triangles,
        Some(SurfaceTable::new(&palette, indices)),
    );
    world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::mesh(floor)).friction(0.5))
            .mass(0.0)
            .position([0.0, 0.0, 0.0]),
    );
    floor
}

fn drop_ball(world: &mut World, position: [f32; 3]) -> dynamis_model::BodyHandle {
    world.spawn(
        BodyDesc::sphere(0.3)
            .mass(1.0)
            .position(position)
            .friction(0.5),
    )
}

#[test]
fn a_triangle_surface_supplies_the_contact_material() {
    let mut world = observed_world(gravity_config());
    flat_mesh(&mut world, 0.0, &[0]);
    drop_ball(&mut world, [0.5, 0.31, -1.0]);
    settle(&mut world, 8);
    let manifolds = world.inspect_contacts();
    let manifold = manifolds
        .first()
        .expect("a resting ball must report its contact");
    assert_eq!(
        deepest(manifold).triangle,
        Some(0),
        "the ball rests on the only triangle"
    );
    assert_eq!(
        manifold
            .surface
            .expect("the hit triangle carries a surface")
            .friction,
        ICE
    );
    let expected = (ICE * 0.5f32).sqrt();
    assert!(
        (manifold.material.friction - expected).abs() < 1e-6,
        "the contact must combine the triangle surface with the collider, got {}",
        manifold.material.friction
    );
}

#[test]
fn each_triangle_carries_its_own_surface() {
    let mut world = observed_world(static_config());
    let (vertices, triangles) = two_patches();
    let palette = palette();
    let mesh = world.add_mesh(
        &vertices,
        &triangles,
        Some(SurfaceTable::new(&palette, &[0, 1])),
    );
    world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::mesh(mesh)))
            .mass(0.0)
            .position([0.0, 0.0, 0.0]),
    );
    let low = world.ray_query(
        [0.5, 5.0, -1.0],
        [0.0, -1.0, 0.0],
        20.0,
        &QueryFilter::default(),
    );
    let high = world.ray_query(
        [8.5, 5.0, -1.0],
        [0.0, -1.0, 0.0],
        20.0,
        &QueryFilter::default(),
    );
    world.step(DT);
    world.wait();
    let low = world
        .query_hit(low)
        .expect("the ray must hit the low patch");
    assert_eq!(low.triangle(), Some(0));
    assert_eq!(
        low.surface()
            .expect("the low patch carries a surface")
            .friction,
        ICE
    );
    let high = world
        .query_hit(high)
        .expect("the ray must hit the high patch");
    assert_eq!(high.triangle(), Some(1));
    assert_eq!(
        high.surface()
            .expect("the high patch carries a surface")
            .friction,
        ROCK
    );
}

#[test]
fn the_deepest_triangle_supplies_the_manifold_material() {
    let mut world = observed_world(gravity_config());
    let (vertices, triangles) = two_patches();
    let palette = palette();
    let mesh = world.add_mesh(
        &vertices,
        &triangles,
        Some(SurfaceTable::new(&palette, &[0, 1])),
    );
    world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::mesh(mesh)).friction(0.5))
            .mass(0.0)
            .position([0.0, 0.0, 0.0]),
    );
    drop_ball(&mut world, [0.5, 0.31, -1.0]);
    settle(&mut world, 8);
    let manifolds = world.inspect_contacts();
    let manifold = manifolds
        .first()
        .expect("a resting ball must report its contact");
    assert_eq!(deepest(manifold).triangle, Some(0));
    assert_eq!(
        manifold
            .surface
            .expect("the low patch carries a surface")
            .friction,
        ICE
    );
}

#[test]
fn an_unsurfaced_mesh_keeps_the_collider_material() {
    let mut world = observed_world(gravity_config());
    let vertices = patch(0.0, 0.0);
    let triangles = vec![[0u32, 1, 2]];
    let floor = world.add_mesh(&vertices, &triangles, None);
    world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::mesh(floor)).friction(0.7))
            .mass(0.0)
            .position([0.0, 0.0, 0.0]),
    );
    drop_ball(&mut world, [0.5, 0.31, -1.0]);
    settle(&mut world, 8);
    let manifolds = world.inspect_contacts();
    let manifold = manifolds
        .first()
        .expect("a resting ball must report its contact");
    assert!(
        manifold.surface.is_none(),
        "a mesh without surfaces reports no surface"
    );
    let expected = (0.7 * 0.5f32).sqrt();
    assert!(
        (manifold.material.friction - expected).abs() < 1e-6,
        "an unsurfaced mesh keeps the collider material, got {}",
        manifold.material.friction
    );
}

#[test]
fn a_height_field_carries_one_surface_per_cell() {
    let mut world = observed_world(gravity_config());
    let heights = [0.0f32, 0.0, 0.0, 0.0, 0.0, 1.0];
    let palette = palette();
    let field = world.add_height_field(
        2,
        3,
        &heights,
        [6.0, 6.0],
        Some(SurfaceTable::new(&palette, &[0, 1])),
    );
    world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::height_field(field)).friction(0.5))
            .mass(0.0)
            .position([0.0, 0.0, 0.0]),
    );
    drop_ball(&mut world, [2.0, 0.31, 2.0]);
    settle(&mut world, 8);
    let manifolds = world.inspect_contacts();
    let manifold = manifolds
        .first()
        .expect("a resting ball must report its contact");
    assert!(
        matches!(deepest(manifold).triangle, Some(0) | Some(1)),
        "the ball rests inside the first height field cell, got {:?}",
        deepest(manifold).triangle
    );
    assert_eq!(
        manifold
            .surface
            .expect("the height field cell carries a surface")
            .friction,
        ICE
    );
}

#[test]
fn a_convex_hit_reports_no_triangle() {
    let mut world = observed_world(static_config());
    world.spawn(BodyDesc::sphere(0.5).mass(0.0).position([0.0, 0.0, 0.0]));
    let query = world.ray_query(
        [0.0, 5.0, 0.0],
        [0.0, -1.0, 0.0],
        20.0,
        &QueryFilter::default(),
    );
    world.step(DT);
    world.wait();
    let hit = world.query_hit(query).expect("the ray must hit the sphere");
    assert!(hit.surface().is_none() && hit.triangle().is_none());
}

#[test]
fn surface_resolution_survives_a_snapshot() {
    let mut world = observed_world(gravity_config());
    flat_mesh(&mut world, 0.0, &[0]);
    drop_ball(&mut world, [0.5, 0.31, -1.0]);
    settle(&mut world, 8);
    let snapshot = world.snapshot();
    settle(&mut world, 4);
    world.restore(&snapshot);
    settle(&mut world, 4);
    let manifolds = world.inspect_contacts();
    let manifold = manifolds
        .first()
        .expect("a restored ball must still touch its floor");
    assert_eq!(
        manifold
            .surface
            .expect("the restored mesh keeps its surfaces")
            .friction,
        ICE
    );
}

#[test]
fn an_out_of_range_surface_index_is_refused() {
    let mut world = observed_world(static_config());
    let vertices = patch(0.0, 0.0);
    let triangles = vec![[0u32, 1, 2]];
    let palette = palette();
    let refined = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        SurfaceTable::new(&palette, &[2])
    }));
    assert!(
        refined.is_err(),
        "a surface index beyond its palette must be refused"
    );
    let (palette, indices) = (palette.to_vec(), vec![0u32, 0]);
    let mismatched = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        world.add_mesh(
            &vertices,
            &triangles,
            Some(SurfaceTable::new(&palette, &indices)),
        )
    }));
    assert!(
        mismatched.is_err(),
        "a surface table must carry one surface per triangle"
    );
}

#[test]
fn a_height_field_refuses_a_surface_table_of_the_wrong_size() {
    let mut world = observed_world(static_config());
    let palette = palette();
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        world.add_height_field(
            2,
            3,
            &[0.0; 6],
            [6.0, 6.0],
            Some(SurfaceTable::new(&palette, &[0, 1, 0])),
        )
    }));
    assert!(
        refused.is_err(),
        "a height field carries one surface per cell"
    );
}

#[test]
fn a_hull_refuses_a_surface_table() {
    let mut world = observed_world(static_config());
    let vertices = vec![
        [0.0f32, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    let triangles = vec![
        [0u32, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [3, 7, 6],
        [3, 6, 2],
        [0, 4, 7],
        [0, 7, 3],
        [1, 2, 6],
        [1, 6, 5],
    ];
    let hull = world.add_hull(&vertices, &triangles);
    let palette = palette();
    let indices = vec![0u32; triangles.len()];
    let refused = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        world.update_mesh(
            hull,
            &vertices,
            &triangles,
            Some(SurfaceTable::new(&palette, &indices)),
        )
    }));
    assert!(
        refused.is_err(),
        "a hull has no addressable faces for surfaces"
    );
}

#[test]
fn a_reshaped_mesh_replaces_its_surfaces() {
    let mut world = observed_world(gravity_config());
    let floor = flat_mesh(&mut world, 0.0, &[0]);
    let ball = drop_ball(&mut world, [0.5, 0.31, -1.0]);
    settle(&mut world, 8);
    let vertices = patch(0.0, 0.0);
    let triangles = vec![[0u32, 1, 2]];
    let palette = [rock()];
    world.update_mesh(
        floor,
        &vertices,
        &triangles,
        Some(SurfaceTable::new(&palette, &[0])),
    );
    world.wake(ball);
    settle(&mut world, 8);
    let manifolds = world.inspect_contacts();
    let manifold = manifolds
        .first()
        .expect("the ball must still touch the reshaped floor");
    assert_eq!(
        manifold
            .surface
            .expect("the reshaped floor carries a surface")
            .friction,
        ROCK
    );
}
