use super::common::{DT, asleep, gravity_config, settle_until, sim, static_config};
use dynamis_model::{BodyDesc, ColliderDesc, MaterialCombine, PhysicsConfig, QueryFilter, Shape};

#[test]
fn friction_combine_modes_scale_grip() {
    let mut grip = sim(gravity_config());
    let ball = grip.spawn(BodyDesc::sphere(0.5).position([0.0, 2.0, 0.0]));
    let _floor = grip.spawn(
        BodyDesc::new(
            ColliderDesc::new(Shape::cuboid([5.0, 0.5, 5.0]))
                .friction(1.0)
                .restitution(0.0),
        )
        .position([0.0, -0.5, 0.0])
        .mass(0.0),
    );
    settle_until(&mut grip, 300, |world| asleep(world));
    let grip_state = grip.read_state(ball);
    assert!(
        (grip_state.position[1] - 0.5).abs() < 0.02 && grip_state.velocity[0].abs() < 0.05,
        "high friction must hold the ball at rest, got {:?} {:?}",
        grip_state.position,
        grip_state.velocity
    );

    let mut slick = sim(PhysicsConfig {
        friction_combine: MaterialCombine::Min,
        ..gravity_config()
    });
    let ball = slick.spawn(BodyDesc::sphere(0.5).position([0.0, 2.0, 0.0]));
    let _floor = slick.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::cuboid([5.0, 0.5, 5.0])).friction(0.1))
            .position([0.0, -0.5, 0.0])
            .mass(0.0),
    );
    settle_until(&mut slick, 300, |world| {
        world.read_state(ball).position[1] < 0.6
    });
    let slick_state = slick.read_state(ball);
    assert!(
        slick_state.position[1] > 0.45,
        "min-combined friction must yield zero grip friction, got {:?}",
        slick_state.position
    );
}

#[test]
fn set_friction_updates_collider_material() {
    let mut world = sim(gravity_config());
    let ball = world.spawn(BodyDesc::sphere(0.5).position([0.0, 0.45, 0.0]));
    let _floor = world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::cuboid([5.0, 0.5, 5.0])))
            .position([0.0, -0.5, 0.0])
            .mass(0.0),
    );
    world.set_friction(ball, 0.0);
    settle_until(&mut world, 300, |world| asleep(world));
    let state = world.read_state(ball);
    assert!(
        (state.position[1] - 0.5).abs() < 0.05,
        "zero friction ball must stay at rest height, got {:?}",
        state.position
    );
}

#[test]
fn plane_floor_supports_resting_contact() {
    let mut world = sim(gravity_config());
    let ball = world.spawn(BodyDesc::sphere(0.5).position([0.0, 0.45, 0.0]));
    let _plane = world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::plane()))
            .position([0.0, 0.0, 0.0])
            .mass(0.0),
    );
    settle_until(&mut world, 240, |world| asleep(world));
    let state = world.read_state(ball);
    assert!(
        (state.position[1] - 0.5).abs() < 0.02,
        "ball must rest on the plane, got {:?}",
        state.position
    );
}

#[test]
fn tilted_plane_keeps_contact_normal() {
    let mut world = sim(gravity_config());
    let ball = world.spawn(BodyDesc::sphere(0.5).position([0.0, 0.8, 0.0]));
    let _plane = world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::plane()).rotation([
            0.0,
            (std::f32::consts::PI / 12.0).sin(),
            0.0,
            (std::f32::consts::PI / 12.0).cos(),
        ]))
        .position([0.0, 0.0, 0.0])
        .mass(0.0),
    );
    settle_until(&mut world, 200, |world| {
        let state = world.read_state(ball);
        state.position[1] < 0.8 && state.velocity[1].abs() < 0.1
    });
    let state = world.read_state(ball);
    let height = state.position[1];
    assert!(
        height > 0.2 && height < 0.8,
        "ball must rest on the inclined plane surface, got y={height}"
    );
}

#[test]
fn ray_hits_plane_and_reports_surface() {
    let mut world = sim(static_config());
    let plane = world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::plane()))
            .position([0.0, 0.0, 0.0])
            .mass(0.0),
    );
    let query = world.ray_query(
        [0.0, 2.0, 0.0],
        [0.0, -1.0, 0.0],
        10.0,
        &QueryFilter::default(),
    );
    world.step(DT);
    world.wait();
    let hit = world.query_hit(query).expect("ray must hit the plane");
    assert_eq!(hit.body, plane);
    assert!((hit.distance - 2.0).abs() < 1e-3);
    assert!((hit.point[1] - 0.0).abs() < 1e-3);
    assert!((hit.normal[1] - 1.0).abs() < 1e-3);
}

#[test]
fn sweep_over_plane_stops_at_surface() {
    let mut world = sim(static_config());
    let plane = world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::plane()))
            .position([0.0, 0.0, 0.0])
            .mass(0.0),
    );
    let query = world.sweep_query(
        &Shape::sphere(0.4),
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 2.0, 0.0],
        [0.0, -1.0, 0.0],
        10.0,
        &QueryFilter::default(),
    );
    world.step(DT);
    world.wait();
    let hit = world.query_hit(query).expect("sweep must hit the plane");
    assert_eq!(hit.body, plane);
    assert!(
        (hit.distance - 1.6).abs() < 1e-3,
        "sweep must stop at the plane surface, got {}",
        hit.distance
    );
}

#[test]
fn scaled_cuboid_collides_at_scaled_extent() {
    let mut world = sim(static_config());
    let ball = world.spawn(
        BodyDesc::sphere(0.5)
            .position([-8.0, 0.0, 0.0])
            .velocity([6.0, 0.0, 0.0]),
    );
    let _box = world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::cuboid([1.0, 1.0, 1.0])).scale([4.0, 1.0, 1.0]))
            .position([0.0, 0.0, 0.0])
            .mass(0.0),
    );
    settle_until(&mut world, 240, |world| {
        let state = world.read_state(ball);
        state.position[0] > -5.0 && state.velocity[0].abs() < 0.05
    });
    let state = world.read_state(ball);
    assert!(
        state.position[0] > -5.0 && state.position[0] < -4.2,
        "scaled box must block the ball at x=-4.5, got {:?}",
        state.position
    );
}

#[test]
fn scaled_mesh_ray_hit_uses_local_scale() {
    let mut world = sim(static_config());
    let vertices = vec![
        [-1.0f32, 0.0, -1.0],
        [1.0, 0.0, -1.0],
        [1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
    ];
    let triangles = vec![[0u32, 2, 1], [0, 3, 2]];
    let floor = world.add_mesh(&vertices, &triangles);
    let _body = world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::mesh(floor)).scale([1.0, 2.0, 1.0]))
            .position([0.0, 0.0, 0.0])
            .mass(0.0),
    );
    let query = world.ray_query(
        [0.0, 3.0, 0.0],
        [0.0, -1.0, 0.0],
        10.0,
        &QueryFilter::default(),
    );
    world.step(DT);
    world.wait();
    let hit = world
        .query_hit(query)
        .expect("ray must hit the scaled mesh");
    assert!(
        (hit.distance - 3.0).abs() < 1e-3,
        "scaled mesh must double its height along y, got {}",
        hit.distance
    );
}

#[test]
fn plane_ccd_stops_fast_ball() {
    let mut world = sim(static_config());
    let _plane = world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::plane()))
            .position([0.0, 0.0, 0.0])
            .mass(0.0),
    );
    let ball = world.spawn(
        BodyDesc::sphere(0.3)
            .position([0.0, 10.0, 0.0])
            .velocity([0.0, -400.0, 0.0])
            .ccd(true),
    );
    world.step(DT);
    world.wait();
    let state = world.read_state(ball);
    assert!(
        state.position[1] > 0.2,
        "ccd must stop the fast ball above the plane, got {:?}",
        state.position
    );
}
