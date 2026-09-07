use super::common::{DT, flat_mesh_floor, settle, sim, static_config};
use dynamis_model::{BodyDesc, ColliderDesc, QueryFilter, Shape};
use dynamis_sim::Simulation;

struct RayCase {
    name: &'static str,
    build: fn(&mut Simulation) -> (dynamis_model::BodyHandle, f32),
    tolerance: f32,
}

fn ray_sphere(world: &mut Simulation) -> (dynamis_model::BodyHandle, f32) {
    let body = world.spawn(BodyDesc::sphere(0.5).position([0.0, 0.0, 5.0]));
    (body, 4.5)
}

fn ray_box(world: &mut Simulation) -> (dynamis_model::BodyHandle, f32) {
    let body = world.spawn(BodyDesc::cuboid([1.0, 1.0, 1.0]).position([0.0, 0.0, 5.0]));
    (body, 4.0)
}

fn ray_capsule(world: &mut Simulation) -> (dynamis_model::BodyHandle, f32) {
    let body = world.spawn(BodyDesc::capsule(0.5, 1.0).position([0.0, 0.0, 5.0]));
    (body, 4.5)
}

fn ray_cylinder(world: &mut Simulation) -> (dynamis_model::BodyHandle, f32) {
    let body = world.spawn(BodyDesc::cylinder(0.5, 1.0).position([0.0, 0.0, 5.0]));
    (body, 4.5)
}

fn ray_hull(world: &mut Simulation) -> (dynamis_model::BodyHandle, f32) {
    let vertices = vec![
        [-1.0f32, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ];
    let triangles = vec![
        [0u32, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [1, 5, 4],
        [1, 4, 0],
        [2, 6, 5],
        [2, 5, 1],
        [3, 7, 6],
        [3, 6, 2],
        [0, 4, 7],
        [0, 7, 3],
    ];
    let source = world.add_hull(&vertices, &triangles);
    let body = world
        .spawn(BodyDesc::new(ColliderDesc::new(Shape::hull(source))).position([0.0, 0.0, 5.0]));
    (body, 4.0)
}

#[test]
fn raycast_reaches_every_convex_shape_exactly() {
    let cases = [
        RayCase {
            name: "sphere",
            build: ray_sphere,
            tolerance: 1e-3,
        },
        RayCase {
            name: "box",
            build: ray_box,
            tolerance: 1e-3,
        },
        RayCase {
            name: "capsule",
            build: ray_capsule,
            tolerance: 1e-3,
        },
        RayCase {
            name: "cylinder",
            build: ray_cylinder,
            tolerance: 1e-3,
        },
        RayCase {
            name: "hull",
            build: ray_hull,
            tolerance: 5e-3,
        },
    ];
    for case in cases {
        let mut world = sim(8, static_config());
        let (body, expected) = (case.build)(&mut world);
        let query = world.ray_query(
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            20.0,
            &QueryFilter::default(),
        );
        world.step(DT);
        world.wait();
        let hit = world.query_hit(query).expect(case.name);
        assert_eq!(hit.body, body, "{}", case.name);
        assert!(
            (hit.distance - expected).abs() < case.tolerance,
            "{}: expected {expected}, got {}",
            case.name,
            hit.distance
        );
    }
}

#[test]
fn mesh_floor_catches_ball_and_blocks_from_below() {
    let mut world = sim(8, super::common::gravity_config());
    flat_mesh_floor(&mut world);
    let ball = world.spawn(BodyDesc::sphere(0.5).position([0.0, 5.0, 0.0]));
    settle(&mut world, 120);
    let y = world.read_state(ball).position[1];
    assert!(
        (y - 0.5).abs() < 0.05,
        "ball must rest on the mesh floor, got {y}"
    );

    let mut world = sim(8, super::common::gravity_config());
    flat_mesh_floor(&mut world);
    let ball = world.spawn(
        BodyDesc::sphere(0.5)
            .position([0.0, -3.0, 0.0])
            .velocity([0.0, 10.0, 0.0])
            .restitution(0.0),
    );
    let mut peak = f32::MIN;
    for _ in 0..40 {
        world.step(DT);
        world.wait();
        peak = peak.max(world.read_state(ball).position[1]);
    }
    assert!(
        peak <= -0.45,
        "mesh plane must block the body from below, peak {peak}"
    );
}

#[test]
fn height_field_catches_ball_and_blocks_from_below() {
    let mut world = sim(8, super::common::gravity_config());
    let heights = vec![0.0f32; 9];
    let source = world.add_height_field(3, 3, &heights, [2.0, 2.0]);
    world.spawn(BodyDesc::new(ColliderDesc::new(Shape::height_field(source))).mass(0.0));
    let ball = world.spawn(BodyDesc::sphere(0.5).position([1.0, 5.0, 1.0]));
    settle(&mut world, 120);
    let y = world.read_state(ball).position[1];
    assert!(
        (y - 0.5).abs() < 0.05,
        "ball must rest on the field, got {y}"
    );

    let mut world = sim(8, super::common::gravity_config());
    let heights = vec![0.0f32; 9];
    let source = world.add_height_field(3, 3, &heights, [2.0, 2.0]);
    world.spawn(BodyDesc::new(ColliderDesc::new(Shape::height_field(source))).mass(0.0));
    let ball = world.spawn(
        BodyDesc::sphere(0.5)
            .position([0.0, -3.0, 0.0])
            .velocity([0.0, 10.0, 0.0])
            .restitution(0.0),
    );
    let mut peak = f32::MIN;
    for _ in 0..40 {
        world.step(DT);
        world.wait();
        peak = peak.max(world.read_state(ball).position[1]);
    }
    assert!(
        peak <= -0.45,
        "height field must block the body from below, peak {peak}"
    );
}

#[test]
fn height_field_ramp_directs_ball_downhill() {
    let mut world = sim(8, super::common::gravity_config());
    let mut heights = Vec::new();
    for _row in 0..3 {
        for col in 0..5 {
            heights.push((4 - col) as f32);
        }
    }
    let source = world.add_height_field(3, 5, &heights, [2.0, 2.0]);
    world.spawn(BodyDesc::new(ColliderDesc::new(Shape::height_field(source))).mass(0.0));
    let ball = world.spawn(
        BodyDesc::sphere(0.3)
            .position([0.5, 5.0, 2.0])
            .restitution(0.0),
    );
    settle(&mut world, 150);
    let state = world.read_state(ball);
    assert!(
        state.position[0] > 4.0,
        "ball must roll downhill along the descending x axis, got {:?}",
        state.position
    );
}

#[test]
fn cylinder_rests_at_half_height() {
    let mut world = sim(8, super::common::gravity_config());
    flat_mesh_floor(&mut world);
    let cylinder = world.spawn(
        BodyDesc::cylinder(0.5, 1.0)
            .position([0.0, 3.0, 0.0])
            .restitution(0.0),
    );
    settle(&mut world, 120);
    let y = world.read_state(cylinder).position[1];
    assert!(
        (y - 1.0).abs() < 0.05,
        "cylinder must rest at y=1.0, got {y}"
    );
}

#[test]
fn capsule_rests_upright_on_flat_floor() {
    let mut world = sim(8, super::common::gravity_config());
    flat_mesh_floor(&mut world);
    let capsule = world.spawn(
        BodyDesc::capsule(0.3, 1.0)
            .position([0.0, 3.0, 0.0])
            .restitution(0.0),
    );
    settle(&mut world, 150);
    let y = world.read_state(capsule).position[1];
    assert!(
        (y - 1.3).abs() < 0.1,
        "capsule must rest on its bottom sphere at y=1.3, got {y}"
    );
}

#[test]
fn box_rests_flat_on_static_ground() {
    let mut world = sim(8, super::common::gravity_config());
    world.spawn(
        BodyDesc::cuboid([20.0, 1.0, 20.0])
            .mass(0.0)
            .position([0.0, -1.0, 0.0]),
    );
    let cube = world.spawn(
        BodyDesc::cuboid([0.5, 0.5, 0.5])
            .position([0.0, 3.0, 0.0])
            .restitution(0.0),
    );
    settle(&mut world, 150);
    let y = world.read_state(cube).position[1];
    assert!((y - 0.5).abs() < 0.05, "cube must rest at y=0.5, got {y}");
}

#[test]
fn compound_body_rests_on_lowest_child() {
    let mut world = sim(8, super::common::gravity_config());
    flat_mesh_floor(&mut world);
    let compound = world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::sphere(0.5)))
            .collider(ColliderDesc::new(Shape::sphere(0.5)).offset([0.0, 0.8, 0.0]))
            .position([0.0, 3.0, 0.0])
            .restitution(0.0),
    );
    settle(&mut world, 120);
    let y = world.read_state(compound).position[1];
    assert!(
        (y - 0.5).abs() < 0.05,
        "compound must rest on its lowest child, got {y}"
    );
}

#[test]
fn collider_offset_shifts_hit_surface() {
    let mut world = sim(8, static_config());
    let body = world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::sphere(0.5)).offset([0.0, 0.0, -0.5]))
            .position([0.0, 0.0, 5.0]),
    );
    let query = world.ray_query(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        20.0,
        &QueryFilter::default(),
    );
    world.step(DT);
    world.wait();
    let hit = world.query_hit(query).expect("offset collider must be hit");
    assert_eq!(hit.body, body);
    assert!(
        (hit.distance - 4.0).abs() < 1e-3,
        "offset must move the surface to 4.0, got {}",
        hit.distance
    );
}

#[test]
fn collider_rotation_reshapes_hit_geometry() {
    let sin_half = std::f32::consts::FRAC_PI_8.sin();
    let cos_half = std::f32::consts::FRAC_PI_8.cos();
    let half_sqrt_two: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let mut world = sim(8, static_config());
    let body = world.spawn(
        BodyDesc::new(
            ColliderDesc::new(Shape::cuboid([0.5, 0.5, 0.5]))
                .rotation([0.0, sin_half, 0.0, cos_half]),
        )
        .position([0.0, 0.0, 5.0]),
    );
    let query = world.ray_query(
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        20.0,
        &QueryFilter::default(),
    );
    world.step(DT);
    world.wait();
    let hit = world
        .query_hit(query)
        .expect("rotated collider must be hit");
    assert_eq!(hit.body, body);
    let expected = 5.0 - 2.0 * 0.5 * half_sqrt_two;
    assert!(
        (hit.distance - expected).abs() < 2e-3,
        "45-degree spin must extend the half width to sqrt(0.5), got {}",
        hit.distance
    );
}

#[test]
fn set_collider_and_set_shape_replace_geometry() {
    let mut world = sim(8, static_config());
    let body = world.spawn(BodyDesc::sphere(0.3));
    world.set_collider(body, 0, ColliderDesc::new(Shape::sphere(0.9)));
    world.step(DT);
    world.wait();
    let query = world.ray_query(
        [0.0, 2.0, 0.0],
        [0.0, -1.0, 0.0],
        10.0,
        &QueryFilter::default(),
    );
    world.step(DT);
    world.wait();
    let hit = world
        .query_hit(query)
        .expect("enlarged collider must be hit");
    assert_eq!(hit.body, body);
    assert!(
        (hit.distance - 1.1).abs() < 1e-3,
        "enlarged collider must be reachable at 1.1, got {}",
        hit.distance
    );

    let mut world = sim(8, super::common::gravity_config());
    world.spawn(
        BodyDesc::cuboid([20.0, 1.0, 20.0])
            .mass(0.0)
            .position([0.0, -1.0, 0.0]),
    );
    let switched = world.spawn(
        BodyDesc::sphere(0.5)
            .position([0.0, 3.0, 0.0])
            .restitution(0.0),
    );
    world.set_shape(switched, Shape::cuboid([0.5, 0.5, 0.5]));
    settle(&mut world, 150);
    let y = world.read_state(switched).position[1];
    assert!(
        (y - 0.5).abs() < 0.05,
        "swapped box must rest flat at y=0.5, got {y}"
    );
}

#[test]
fn hull_cube_rests_on_floor() {
    let mut world = sim(8, super::common::gravity_config());
    let vertices = vec![
        [-1.0f32, -1.0, -1.0],
        [1.0, -1.0, -1.0],
        [1.0, 1.0, -1.0],
        [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0],
        [1.0, -1.0, 1.0],
        [1.0, 1.0, 1.0],
        [-1.0, 1.0, 1.0],
    ];
    let triangles = vec![
        [0u32, 1, 2],
        [0, 2, 3],
        [4, 6, 5],
        [4, 7, 6],
        [0, 4, 5],
        [0, 5, 1],
        [1, 5, 6],
        [1, 6, 2],
        [2, 6, 7],
        [2, 7, 3],
        [3, 7, 4],
        [3, 4, 0],
    ];
    let source = world.add_hull(&vertices, &triangles);
    flat_mesh_floor(&mut world);
    let hull = world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::hull(source)))
            .position([0.0, 3.0, 0.0])
            .restitution(0.0),
    );
    settle(&mut world, 150);
    let y = world.read_state(hull).position[1];
    assert!(
        (y - 1.0).abs() < 0.05,
        "hull cube must rest at y=1.0, got {y}"
    );
}

#[test]
fn penetration_solver_expels_embedded_hull() {
    let mut world = sim(8, super::common::gravity_config());
    world.spawn(
        BodyDesc::cuboid([20.0, 1.0, 20.0])
            .mass(0.0)
            .position([0.0, -1.0, 0.0]),
    );
    let vertices = vec![
        [-0.5f32, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, -0.5, 0.5],
        [-0.5, -0.5, 0.5],
        [-0.5, 0.5, -0.5],
        [0.5, 0.5, -0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ];
    let triangles = vec![
        [0u32, 1, 2],
        [0, 2, 3],
        [4, 6, 5],
        [4, 7, 6],
        [0, 4, 5],
        [0, 5, 1],
        [1, 5, 6],
        [1, 6, 2],
        [2, 6, 7],
        [2, 7, 3],
        [3, 7, 4],
        [3, 4, 0],
    ];
    let source = world.add_hull(&vertices, &triangles);
    let hull = world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::hull(source)))
            .position([0.0, 0.3, 0.0])
            .restitution(0.0),
    );
    settle(&mut world, 90);
    let y = world.read_state(hull).position[1];
    assert!(
        y > 0.3 && y < 2.0,
        "embedded hull must be expelled to the box top, got {y}"
    );
}
