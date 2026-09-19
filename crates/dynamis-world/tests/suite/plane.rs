use super::common::{asleep, gravity_config, observed_world, settle_until};
use dynamis_model::{BodyDesc, BodyHandle, ColliderDesc, Shape};
use dynamis_world::{ContactManifold, World};

const SLOP: f32 = 0.005;

fn plane_at(world: &mut World, position: [f32; 3], orientation: [f32; 4]) -> BodyHandle {
    world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::plane()))
            .position(position)
            .orientation(orientation)
            .mass(0.0),
    )
}

fn cube(extent: f32) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let half = extent / 2.0;
    let vertices = vec![
        [-half, -half, -half],
        [half, -half, -half],
        [half, half, -half],
        [-half, half, -half],
        [-half, -half, half],
        [half, -half, half],
        [half, half, half],
        [-half, half, half],
    ];
    let triangles = vec![
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [3, 7, 6],
        [3, 6, 2],
        [1, 2, 6],
        [1, 6, 5],
        [0, 4, 7],
        [0, 7, 3],
    ];
    (vertices, triangles)
}

fn hull_cube(world: &mut World) -> (Shape, f32) {
    let (vertices, triangles) = cube(1.0);
    let shape = Shape::hull(world.add_hull_from_mesh(&vertices, &triangles));
    (shape, 0.5)
}

fn rests_on_a_plane(shape: Shape, hull_half_extent: f32) -> Option<f32> {
    match shape {
        Shape::Sphere { radius } => Some(radius),
        Shape::Cuboid { half_extents } => Some(half_extents[1]),
        Shape::Capsule {
            radius,
            half_height,
        } => Some(radius + half_height),
        Shape::Cylinder { half_height, .. } => Some(half_height),
        Shape::Hull(_) => Some(hull_half_extent),
        Shape::Mesh(_) | Shape::HeightField(_) | Shape::Plane => None,
    }
}

fn primitives(world: &mut World) -> Vec<(&'static str, Shape, f32)> {
    let (hull, hull_half_extent) = hull_cube(world);
    [
        ("sphere", Shape::sphere(0.5)),
        ("cuboid", Shape::cuboid([0.5; 3])),
        ("capsule", Shape::capsule(0.5, 0.5)),
        ("cylinder", Shape::cylinder(0.5, 0.5)),
        ("hull", hull),
    ]
    .into_iter()
    .map(|(label, shape)| {
        let height = rests_on_a_plane(shape, hull_half_extent)
            .expect("every shape a simulating body may carry must rest on a plane");
        (label, shape, height - SLOP)
    })
    .collect()
}

fn settle_on_plane(world: &mut World) {
    settle_until(world, 480, |world| asleep(world));
}

fn assert_rests(label: &'static str, world: &World, body: BodyHandle, rest_height: f32) {
    let state = world.read_state(body);
    assert!(
        (state.position[1] - rest_height).abs() < 0.02,
        "a {label} must rest on a plane at {rest_height}, got {:?}",
        state.position,
    );
    assert!(
        state.velocity[1].abs() < 0.05
            && state.velocity[0].abs() < 0.05
            && state.velocity[2].abs() < 0.05,
        "a resting {label} must hold still, got velocity {:?}",
        state.velocity,
    );
}

fn manifolds_of(world: &mut World, body: BodyHandle) -> Vec<ContactManifold> {
    world
        .inspect_contacts()
        .into_iter()
        .filter(|manifold| manifold.first == body || manifold.second == body)
        .collect()
}

#[test]
fn every_solid_primitive_rests_on_a_plane() {
    let mut world = observed_world(gravity_config());
    plane_at(&mut world, [0.0; 3], [0.0, 0.0, 0.0, 1.0]);
    let primitives = primitives(&mut world);
    let bodies = primitives
        .iter()
        .enumerate()
        .map(|(slot, (_, shape, rest_height))| {
            world.spawn(BodyDesc::new(ColliderDesc::new(*shape)).position([
                slot as f32 * 2.0,
                rest_height + 0.5,
                0.0,
            ]))
        })
        .collect::<Vec<_>>();
    settle_on_plane(&mut world);
    for ((label, _, rest_height), body) in primitives.iter().zip(bodies) {
        assert_rests(label, &world, body, *rest_height);
    }
}

#[test]
fn a_plane_carries_a_primitive_dropped_from_height() {
    let mut world = observed_world(gravity_config());
    plane_at(&mut world, [0.0; 3], [0.0, 0.0, 0.0, 1.0]);
    let primitives = primitives(&mut world);
    let bodies = primitives
        .iter()
        .enumerate()
        .map(|(slot, (_, shape, _))| {
            world.spawn(
                BodyDesc::new(ColliderDesc::new(*shape))
                    .position([slot as f32 * 2.0, 6.0, 0.0])
                    .velocity([0.0, -30.0, 0.0]),
            )
        })
        .collect::<Vec<_>>();
    settle_on_plane(&mut world);
    for ((label, _, rest_height), body) in primitives.iter().zip(bodies) {
        assert_rests(label, &world, body, *rest_height);
    }
}

#[test]
fn a_plane_reaches_every_collider_of_a_compound_body() {
    let mut world = observed_world(gravity_config());
    plane_at(&mut world, [0.0; 3], [0.0, 0.0, 0.0, 1.0]);
    let body = world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::cuboid([0.2; 3])).offset([-0.5, 0.0, 0.0]))
            .position([0.0, 0.7, 0.0])
            .collider(ColliderDesc::new(Shape::cuboid([0.2; 3])).offset([0.5, 0.0, 0.0])),
    );
    settle_on_plane(&mut world);
    assert_rests("compound", &world, body, 0.2 - SLOP);
    let manifolds = manifolds_of(&mut world, body);
    assert_eq!(
        manifolds.len(),
        2,
        "every collider of a compound body must answer its own plane contact",
    );
}

#[test]
fn a_cuboid_meets_a_plane_with_its_supporting_face() {
    let mut world = observed_world(gravity_config());
    plane_at(&mut world, [0.0; 3], [0.0, 0.0, 0.0, 1.0]);
    let body = world.spawn(BodyDesc::cuboid([0.5; 3]).position([0.0, 1.0, 0.0]));
    settle_on_plane(&mut world);
    let manifolds = manifolds_of(&mut world, body);
    let points = &manifolds
        .first()
        .expect("a cuboid on a plane must answer a manifold")
        .points;
    assert_eq!(
        points.len(),
        4,
        "a cuboid face on a plane must answer its four supporting corners",
    );
    let extent = |axis: usize| {
        let values = points.iter().map(|point| point.position[axis]);
        values.clone().fold(f32::MIN, f32::max) - values.fold(f32::MAX, f32::min)
    };
    assert!(
        (extent(0) - 1.0).abs() < 0.05 && (extent(2) - 1.0).abs() < 0.05,
        "the supporting corners must span the whole face, got x={} z={}",
        extent(0),
        extent(2),
    );
    for point in points {
        assert!(
            (point.depth - SLOP).abs() < 0.01,
            "a resting corner must hold the slop, got depth {}",
            point.depth,
        );
    }
}

#[test]
fn a_hull_meets_a_plane_with_its_supporting_face() {
    let mut world = observed_world(gravity_config());
    plane_at(&mut world, [0.0; 3], [0.0, 0.0, 0.0, 1.0]);
    let shape = hull_cube(&mut world).0;
    let body = world.spawn(BodyDesc::new(ColliderDesc::new(shape)).position([0.0, 1.0, 0.0]));
    settle_on_plane(&mut world);
    assert_rests("hull", &world, body, 0.5 - SLOP);
    let manifolds = manifolds_of(&mut world, body);
    assert_eq!(
        manifolds
            .first()
            .expect("a hull on a plane must answer a manifold")
            .points
            .len(),
        4,
        "a hull face on a plane must answer its four supporting corners",
    );
}

#[test]
fn a_cylinder_meets_a_plane_with_its_cap_ring() {
    let mut world = observed_world(gravity_config());
    plane_at(&mut world, [0.0; 3], [0.0, 0.0, 0.0, 1.0]);
    let body = world.spawn(BodyDesc::cylinder(0.5, 0.5).position([0.0, 1.0, 0.0]));
    settle_on_plane(&mut world);
    assert_rests("cylinder", &world, body, 0.5 - SLOP);
    let manifolds = manifolds_of(&mut world, body);
    let points = &manifolds
        .first()
        .expect("a cylinder on a plane must answer a manifold")
        .points;
    assert!(
        points.len() >= 3,
        "a cylinder cap on a plane must answer a ring of points, got {}",
        points.len(),
    );
    let distance = |point: &dynamis_world::ContactPoint| {
        (point.position[0] * point.position[0] + point.position[2] * point.position[2]).sqrt()
    };
    for point in points {
        assert!(
            (distance(point) - 0.5).abs() < 0.02,
            "a cap ring point must sit on the rim, got {:?}",
            point.position,
        );
    }
}

#[test]
fn a_scaled_shape_meets_a_plane_at_its_scaled_extent() {
    let mut world = observed_world(gravity_config());
    plane_at(&mut world, [0.0; 3], [0.0, 0.0, 0.0, 1.0]);
    let hull = hull_cube(&mut world).0;
    let scaled_hull = world.spawn(
        BodyDesc::new(ColliderDesc::new(hull).scale([2.0, 1.0, 3.0])).position([0.0, 2.0, 0.0]),
    );
    let scaled_cuboid = world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::cuboid([0.5; 3])).scale([2.0, 1.0, 3.0]))
            .position([4.0, 2.0, 0.0]),
    );
    settle_on_plane(&mut world);
    assert_rests("scaled hull", &world, scaled_hull, 0.5 - SLOP);
    assert_rests("scaled cuboid", &world, scaled_cuboid, 0.5 - SLOP);
}

#[test]
fn an_inclined_plane_holds_a_cuboid_out_of_its_surface() {
    let mut world = observed_world(gravity_config());
    let tilt: f32 = 0.35;
    let normal = [0.0, tilt.cos(), tilt.sin()];
    plane_at(
        &mut world,
        [0.0; 3],
        [(tilt * 0.5).sin(), 0.0, 0.0, (tilt * 0.5).cos()],
    );
    let body = world.spawn(BodyDesc::cuboid([0.5; 3]).position([0.0, 0.53, 0.0]));
    settle_until(&mut world, 480, |world| asleep(world));
    let state = world.read_state(body);
    let corners = [
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ];
    let rotate = |local: [f32; 3]| {
        let q = state.orientation;
        let u = [q[0], q[1], q[2]];
        let s = q[3];
        let spin = [
            u[1] * local[2] - u[2] * local[1],
            u[2] * local[0] - u[0] * local[2],
            u[0] * local[1] - u[1] * local[0],
        ];
        let spin_spin = [
            u[1] * spin[2] - u[2] * spin[1],
            u[2] * spin[0] - u[0] * spin[2],
            u[0] * spin[1] - u[1] * spin[0],
        ];
        [
            local[0] + 2.0 * (s * spin[0] + spin_spin[0]),
            local[1] + 2.0 * (s * spin[1] + spin_spin[1]),
            local[2] + 2.0 * (s * spin[2] + spin_spin[2]),
        ]
    };
    let mut lowest = f32::MAX;
    for corner in corners {
        let offset = rotate(corner);
        let point = [
            state.position[0] + offset[0],
            state.position[1] + offset[1],
            state.position[2] + offset[2],
        ];
        let clearance = point[0] * normal[0] + point[1] * normal[1] + point[2] * normal[2];
        lowest = lowest.min(clearance);
    }
    assert!(
        lowest > -0.02,
        "no corner of a body on an inclined plane may enter its surface, got {lowest}",
    );
    assert!(
        lowest < 0.06,
        "a body on an inclined plane must stay in contact with it, got {lowest}",
    );
    assert!(
        !manifolds_of(&mut world, body).is_empty(),
        "a cuboid resting on an inclined plane must answer a contact",
    );
}
