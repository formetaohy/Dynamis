use super::common::{asleep, gravity_config, new_world, settle_until, static_config};
use dynamis_model::{BodyDesc, ColliderDesc, QueryFilter, Shape};

fn quad() -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let vertices = vec![
        [-2.0f32, 0.0, -2.0],
        [2.0, 0.0, -2.0],
        [2.0, 0.0, 2.0],
        [-2.0, 0.0, 2.0],
    ];
    let triangles = vec![[0u32, 2, 1], [0, 3, 2]];
    (vertices, triangles)
}

fn sphere_hull(rings: u32, segments: u32) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let mut vertices = vec![[0.0f32, 1.0, 0.0], [0.0, -1.0, 0.0]];
    for ring in 1..rings {
        let theta = std::f32::consts::PI * ring as f32 / rings as f32;
        for segment in 0..segments {
            let phi = std::f32::consts::TAU * segment as f32 / segments as f32;
            vertices.push([
                theta.sin() * phi.cos(),
                theta.cos(),
                theta.sin() * phi.sin(),
            ]);
        }
    }
    let top = 0u32;
    let bottom = 1u32;
    let ring_vertex = |ring: u32, segment: u32| 2 + (ring - 1) * segments + segment % segments;
    let mut triangles = Vec::new();
    for segment in 0..segments {
        triangles.push([top, ring_vertex(1, segment), ring_vertex(1, segment + 1)]);
        triangles.push([
            bottom,
            ring_vertex(rings - 1, segment + 1),
            ring_vertex(rings - 1, segment),
        ]);
    }
    for ring in 1..rings - 1 {
        for segment in 0..segments {
            let a = ring_vertex(ring, segment);
            let b = ring_vertex(ring, segment + 1);
            let c = ring_vertex(ring + 1, segment + 1);
            let d = ring_vertex(ring + 1, segment);
            triangles.push([a, b, c]);
            triangles.push([a, c, d]);
        }
    }
    (vertices, triangles)
}

#[test]
fn a_moved_mesh_floor_carries_a_ball() {
    let mut world = new_world(gravity_config());
    let (vertices, triangles) = quad();
    let floor = world.add_mesh(&vertices, &triangles);
    world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::mesh(floor)))
            .mass(0.0)
            .position([10.0, 0.0, 0.0]),
    );
    let ball = world.spawn(BodyDesc::sphere(0.4).position([10.0, 3.0, 0.0]));
    settle_until(&mut world, 240, |world| asleep(world));
    let y = world.read_state(ball).position[1];
    assert!(
        (y - 0.4).abs() < 0.1,
        "a mesh floor must collide at its body pose, got y={y}"
    );
}

#[test]
fn a_rotated_mesh_wall_stops_a_ball() {
    let mut world = new_world(static_config());
    let (vertices, triangles) = quad();
    let wall = world.add_mesh(&vertices, &triangles);
    let half_turn = std::f32::consts::FRAC_1_SQRT_2;
    world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::mesh(wall)))
            .mass(0.0)
            .orientation([0.0, 0.0, half_turn, half_turn]),
    );
    let ball = world.spawn(
        BodyDesc::sphere(0.4)
            .position([2.0, 0.0, 0.0])
            .velocity([-2.0, 0.0, 0.0]),
    );
    settle_until(&mut world, 120, |world| {
        world.read_state(ball).position[0] < 0.5
    });
    let x = world.read_state(ball).position[0];
    assert!(
        (x - 0.4).abs() < 0.15,
        "a rotated mesh wall must stop the ball at its surface, got x={x}"
    );
}

#[test]
fn a_moved_height_field_carries_a_ball() {
    let mut world = new_world(gravity_config());
    let field = world.add_height_field(2, 2, &[0.5, 0.5, 0.5, 0.5], [4.0, 4.0]);
    world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::height_field(field)))
            .mass(0.0)
            .position([10.0, 0.0, 0.0]),
    );
    let ball = world.spawn(BodyDesc::sphere(0.2).position([11.0, 3.0, 1.0]));
    settle_until(&mut world, 240, |world| {
        (world.read_state(ball).position[1] - 0.7).abs() < 0.1
    });
    let y = world.read_state(ball).position[1];
    assert!(
        (y - 0.7).abs() < 0.1,
        "a height field must collide at its body pose, got y={y}"
    );
}

#[test]
fn a_sweep_query_respects_a_moved_mesh() {
    let mut world = new_world(static_config());
    let (vertices, triangles) = quad();
    let floor = world.add_mesh(&vertices, &triangles);
    let body = world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::mesh(floor)))
            .mass(0.0)
            .position([10.0, 0.0, 0.0]),
    );
    let query = world.sweep_query(
        &Shape::sphere(0.4),
        [0.0, 0.0, 0.0, 1.0],
        [10.0, 2.0, 0.0],
        [0.0, -1.0, 0.0],
        10.0,
        &QueryFilter::default(),
    );
    world.flush_queries();
    let hit = world
        .query_hit(query)
        .expect("the sweep must hit the moved mesh");
    assert_eq!(hit.body, body);
    assert!(
        (1.2..=1.6).contains(&hit.distance),
        "the sweep must stop above the moved mesh surface, got {}",
        hit.distance
    );
}

#[test]
fn a_dense_hull_rests_on_a_moved_mesh_floor() {
    let mut world = new_world(gravity_config());
    let (vertices, triangles) = quad();
    let floor = world.add_mesh(&vertices, &triangles);
    world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::mesh(floor)))
            .mass(0.0)
            .position([10.0, 0.0, 0.0]),
    );
    let (sphere_vertices, sphere_triangles) = sphere_hull(12, 24);
    let hull = world.add_hull(&sphere_vertices, &sphere_triangles);
    let ball =
        world.spawn(BodyDesc::new(ColliderDesc::new(Shape::hull(hull))).position([10.0, 3.0, 0.0]));
    settle_until(&mut world, 240, |world| {
        (world.read_state(ball).position[1] - 1.0).abs() < 0.15
    });
    let y = world.read_state(ball).position[1];
    assert!(
        (y - 1.0).abs() < 0.15,
        "a dense hull must rest on a moved mesh floor, got y={y}"
    );
}

#[test]
fn a_scaled_height_field_lifts_its_surface() {
    let mut world = new_world(gravity_config());
    let field = world.add_height_field(2, 2, &[0.5, 0.5, 0.5, 0.5], [4.0, 4.0]);
    let scaled = world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::height_field(field)).scale([2.0, 2.0, 2.0]))
            .mass(0.0)
            .position([10.0, 0.0, 0.0]),
    );
    let ball = world.spawn(BodyDesc::sphere(0.2).position([11.0, 3.0, 1.0]));
    settle_until(&mut world, 240, |world| {
        world.read_state(ball).position[1] < 1.5
    });
    let y = world.read_state(ball).position[1];
    assert!(
        (y - 1.2).abs() < 0.1,
        "a scaled height field must raise its surface, got y={y}"
    );
    let query = world.sweep_query(
        &Shape::sphere(0.1),
        [0.0, 0.0, 0.0, 1.0],
        [11.0, 3.0, 1.0],
        [0.0, -1.0, 0.0],
        5.0,
        &QueryFilter {
            exclude: Some(ball),
            ..QueryFilter::default()
        },
    );
    world.flush_queries();
    let hit = world
        .query_hit(query)
        .expect("the sweep must reach the scaled field");
    assert_eq!(hit.body, scaled);
    assert!(
        hit.distance < 2.1,
        "the sweep must stop above the raised surface, got {}",
        hit.distance
    );
}
