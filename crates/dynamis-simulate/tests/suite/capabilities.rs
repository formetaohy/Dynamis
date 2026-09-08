use super::common::{DT, gravity_config, settle, sim, static_config, static_sphere_ground};
use dynamis_model::{BodyDesc, ColliderDesc, ConstraintDesc, PhysicsConfig, QueryFilter, Shape};
use std::panic::{AssertUnwindSafe, catch_unwind};

#[test]
fn cylinder_stacks_flat_on_cylinder() {
    let mut world = sim(4, gravity_config());
    let base = world.spawn(
        BodyDesc::cylinder(0.5, 0.5)
            .position([0.0, 0.5, 0.0])
            .mass(0.0),
    );
    let top = world.spawn(BodyDesc::cylinder(0.5, 0.5).position([0.0, 1.49, 0.0]));
    settle(&mut world, 180);
    assert!(
        (world.read_state(top).position[1] - 1.5).abs() < 0.12,
        "cylinder must rest flat on cylinder caps, got {}",
        world.read_state(top).position[1]
    );
    assert!(world.read_state(base).position[1] == 0.5);
}

#[test]
fn hull_cube_stacks_stable() {
    let mut world = sim(4, gravity_config());
    let cube = vec![
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ];
    let faces = vec![
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
    let hull = world.add_hull_from_mesh(&cube, &faces);
    let lower = world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::hull(hull)))
            .position([0.0, 0.5, 0.0])
            .mass(0.0),
    );
    let upper =
        world.spawn(BodyDesc::new(ColliderDesc::new(Shape::hull(hull))).position([0.0, 1.49, 0.0]));
    settle(&mut world, 240);
    assert!(
        (world.read_state(upper).position[1] - 1.5).abs() < 0.15,
        "hull cubes must stack face to face, got {}",
        world.read_state(upper).position[1]
    );
    assert!(world.read_state(lower).position[1] == 0.5);
}

#[test]
fn hull_rests_on_mesh_floor() {
    let mut world = sim(4, gravity_config());
    let cube = vec![
        [-0.5, -0.5, -0.5],
        [0.5, -0.5, -0.5],
        [0.5, 0.5, -0.5],
        [-0.5, 0.5, -0.5],
        [-0.5, -0.5, 0.5],
        [0.5, -0.5, 0.5],
        [0.5, 0.5, 0.5],
        [-0.5, 0.5, 0.5],
    ];
    let faces = vec![
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
    let hull = world.add_hull_from_mesh(&cube, &faces);
    let _floor = super::common::flat_mesh_floor(&mut world);
    let body =
        world.spawn(BodyDesc::new(ColliderDesc::new(Shape::hull(hull))).position([0.0, 1.49, 0.0]));
    settle(&mut world, 180);
    assert!(
        (world.read_state(body).position[1] - 0.5).abs() < 0.1,
        "hull must rest on mesh floor, got {}",
        world.read_state(body).position[1]
    );
}

#[test]
fn contact_manifolds_surface_contact_points() {
    let mut world = sim(4, gravity_config());
    static_sphere_ground(&mut world, 1.0);
    let ball = world.spawn(BodyDesc::sphere(0.5).position([0.0, 1.5, 0.0]));
    settle(&mut world, 30);
    let manifolds = world.contact_manifolds();
    let manifold = manifolds
        .iter()
        .find(|manifold| {
            (manifold.first.id == ball.id || manifold.second.id == ball.id) && !manifold.sensor
        })
        .expect("ball must have a contact manifold");
    assert!(
        manifold
            .points
            .iter()
            .any(|point| { point.depth > -0.1 && point.normal_impulse >= 0.0 }),
        "manifold must expose contact points"
    );
    assert!(
        world.overflow() == (0, 0),
        "quiet simulation must not overflow"
    );
}

#[test]
fn constraint_patch_updates_motor_speed() {
    let mut world = sim(4, static_config());
    let base = world.spawn(BodyDesc::sphere(0.4).mass(0.0));
    let arm = world.spawn(BodyDesc::sphere(0.4).position([0.0, 1.0, 0.0]));
    let joint = world.add_constraint(
        base,
        arm,
        ConstraintDesc::revolute([0.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]),
    );
    world.set_motor(joint, 3.0, 100.0);
    for _ in 0..60 {
        world.step(DT);
    }
    world.wait();
    let spin = world.read_state(arm).angular_velocity[2].abs();
    assert!(
        spin > 2.0,
        "patched motor must drive the revolute joint, got {spin}"
    );
}

#[test]
fn constraint_break_emits_handle_event() {
    let mut world = sim(4, gravity_config());
    static_sphere_ground(&mut world, 1.0);
    let anchor = world.spawn(BodyDesc::sphere(0.2).mass(0.0).position([0.0, 3.0, 0.0]));
    let weight = world.spawn(BodyDesc::sphere(0.3).position([0.0, 3.0, 0.0]).mass(50.0));
    let joint = world.add_constraint(
        anchor,
        weight,
        ConstraintDesc::distance([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], 0.5).break_threshold(1.0, 1.0),
    );
    settle(&mut world, 30);
    let broken = world.drain_constraint_breaks();
    assert!(
        broken.contains(&joint),
        "overloaded constraint must report a break"
    );
    assert!(
        !world.constraints().contains(&joint),
        "broken constraint must be removed"
    );
}

#[test]
fn collider_level_filter_controls_collisions() {
    let mut world = sim(4, static_config());
    let first = world.spawn(
        BodyDesc::new(
            ColliderDesc::new(Shape::sphere(0.5))
                .collision_group(0x2)
                .collision_mask(0x2),
        )
        .position([0.0, -0.4, 0.0])
        .collision_group(0xFFFF_FFFF)
        .collision_mask(0xFFFF_FFFF),
    );
    let second = world.spawn(
        BodyDesc::new(
            ColliderDesc::new(Shape::sphere(0.5))
                .collision_group(0x1)
                .collision_mask(0x1),
        )
        .position([0.0, 0.4, 0.0])
        .collision_group(0xFFFF_FFFF)
        .collision_mask(0xFFFF_FFFF),
    );
    settle(&mut world, 16);
    let separation = world.read_state(second).position[1] - world.read_state(first).position[1];
    assert!(
        (separation - 0.8).abs() < 0.05,
        "non-matching collider filters must disable collision, got {separation}"
    );
}

#[test]
fn per_body_damping_overrides_global() {
    let mut world = sim(4, static_config());
    let loose = world.spawn(BodyDesc::sphere(0.2).velocity([10.0, 0.0, 0.0]));
    let damped = world.spawn(
        BodyDesc::sphere(0.2)
            .velocity([10.0, 0.0, 0.0])
            .damping(3.0),
    );
    settle(&mut world, 30);
    let loose = world.read_state(loose).velocity[0];
    let damped = world.read_state(damped).velocity[0];
    assert!(
        damped < loose * 0.5,
        "per-body damping must exceed global damping, loose {loose} damped {damped}"
    );
}

#[test]
fn gravity_scale_zero_ignores_gravity() {
    let mut world = sim(4, gravity_config());
    let floating = world.spawn(
        BodyDesc::sphere(0.2)
            .position([0.0, 10.0, 0.0])
            .gravity_scale(0.0),
    );
    let falling = world.spawn(BodyDesc::sphere(0.2).position([5.0, 10.0, 0.0]));
    settle(&mut world, 30);
    let floating_y = world.read_state(floating).position[1];
    let falling_y = world.read_state(falling).position[1];
    assert!(
        (floating_y - 10.0).abs() < 1e-3,
        "zero gravity scale must freeze the fall, got {floating_y}"
    );
    assert!(
        falling_y < 9.0,
        "reference body must keep falling, got {falling_y}"
    );
}

#[test]
fn density_sets_mass_from_volume() {
    let mut world = sim(4, static_config());
    let body = world.spawn(BodyDesc::sphere(1.0).density(1.0));
    let state = world.read_state(body);
    let sphere_volume = 4.0 / 3.0 * std::f32::consts::PI;
    assert!(
        (state.inverse_mass - 1.0 / sphere_volume).abs() < 1e-3,
        "density must set mass by volume, inverse mass {}",
        state.inverse_mass
    );
}

#[test]
fn rolling_and_spin_friction_damp_rotation() {
    let mut world = sim(4, gravity_config());
    static_sphere_ground(&mut world, 1.0);
    let braked = world.spawn(
        BodyDesc::new(
            ColliderDesc::new(Shape::sphere(0.2))
                .rolling_friction(0.6)
                .spin_friction(0.6),
        )
        .position([0.0, 1.15, 0.0])
        .angular_velocity([0.0, 8.0, 0.0]),
    );
    let bare = world.spawn(
        BodyDesc::sphere(0.2)
            .position([3.0, 1.15, 0.0])
            .angular_velocity([0.0, 8.0, 0.0]),
    );
    settle(&mut world, 90);
    let braked_spin = world.read_state(braked).angular_velocity[1].abs();
    let bare_spin = world.read_state(bare).angular_velocity[1].abs();
    assert!(
        braked_spin < 1.0,
        "spin friction must stop the twist, got {braked_spin}"
    );
    assert!(
        bare_spin > braked_spin + 1.0,
        "untouched spin must outlast the braked one, bare {bare_spin} braked {braked_spin}"
    );
}

#[test]
fn prev_position_tracks_last_step() {
    let mut world = sim(4, gravity_config());
    let ball = world.spawn(BodyDesc::sphere(0.2).position([0.0, 10.0, 0.0]));
    world.step(DT);
    world.wait();
    let first = world.read_state(ball);
    world.step(DT);
    world.wait();
    let second = world.read_state(ball);
    assert_eq!(second.prev_position, first.position);
    let delta = [
        first.prev_position[0],
        first.prev_position[1] - 10.0,
        first.prev_position[2],
    ];
    assert!(
        super::common::distance(delta, [0.0; 3]) < 1e-4,
        "spawn state must seed previous position"
    );
}

#[test]
fn update_drives_substeps_with_interpolation_alpha() {
    let mut world = sim(4, gravity_config());
    let _ball = world.spawn(BodyDesc::sphere(0.2).position([0.0, 10.0, 0.0]));
    world.update(DT * 0.5, DT, 8);
    assert_eq!(world.count(), 1);
    world.update(DT * 0.5, DT, 8);
    world.wait();
    world.update(DT * 2.5, DT, 2);
    world.wait();
    let alpha = world.interpolation_alpha();
    assert!(
        (0.0..1.0).contains(&alpha),
        "interpolation alpha must stay in [0, 1), got {alpha}"
    );
    world.set_time_scale(2.0);
    world.update(DT, DT, 8);
    world.wait();
    assert!(world.interpolation_alpha() < 1.0);
}

#[test]
fn query_filter_excludes_own_body() {
    let mut world = sim(8, static_config());
    let _ground = static_sphere_ground(&mut world, 1.0);
    let hit = world.spawn(BodyDesc::sphere(0.5).position([5.0, 0.0, 5.0]));
    let _miss = world.spawn(BodyDesc::sphere(0.5).position([5.0, 0.0, -5.0]));
    world.step(DT);
    world.wait();
    let filter = QueryFilter {
        exclude: Some(hit),
        ..QueryFilter::default()
    };
    let handle = world.ray_query([5.0, 0.0, 0.0], [0.0, 0.0, 1.0], 100.0, &filter);
    world.flush_queries();
    assert!(
        world.query_hit(handle).is_none(),
        "excluded body must not be hit"
    );
    let filter = QueryFilter::default();
    let handle = world.ray_query([5.0, 0.0, 0.0], [0.0, 0.0, 1.0], 100.0, &filter);
    world.flush_queries();
    assert_eq!(
        world
            .query_hit(handle)
            .map(|result| (result.body, result.collider)),
        Some((hit, 0)),
        "inclusive query must hit the body and report its collider"
    );
}

#[test]
fn remove_shape_invalidates_handle() {
    let mut world = sim(4, static_config());
    let source = world.add_mesh(
        &[[-1.0, 0.0, -1.0], [1.0, 0.0, -1.0], [1.0, 0.0, 1.0]],
        &[[0, 2, 1]],
    );
    world.remove_shape(source);
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            world.spawn(BodyDesc::new(ColliderDesc::new(Shape::mesh(source))));
        }))
        .is_err(),
        "removed shape source must refuse new bodies"
    );
}

#[test]
fn update_height_field_reshapes_terrain() {
    let mut world = sim(4, gravity_config());
    let source = world.add_height_field(2, 2, &[0.5, 0.5, 0.5, 0.5], [4.0, 4.0]);
    let _floor =
        world.spawn(BodyDesc::new(ColliderDesc::new(Shape::height_field(source))).mass(0.0));
    let ball = world.spawn(BodyDesc::sphere(0.2).position([2.0, 3.0, 2.0]));
    settle(&mut world, 60);
    let rest_a = world.read_state(ball).position[1];
    assert!(
        (rest_a - 0.7).abs() < 0.1,
        "ball must rest on height field at 0.5 + radius, got {rest_a}"
    );
    world.update_height_field(source, 2, 2, &[0.1, 0.1, 0.1, 0.1], [4.0, 4.0]);
    settle(&mut world, 60);
    let rest_b = world.read_state(ball).position[1];
    assert!(
        rest_b < rest_a - 0.2,
        "lowered terrain must lower the rest pose, {rest_a} -> {rest_b}"
    );
}

#[test]
fn sleep_thresholds_override_global_speed() {
    let config = PhysicsConfig {
        sleep_time: 0.05,
        ..static_config()
    };
    let mut world = sim(4, config);
    let lazy = world.spawn(
        BodyDesc::sphere(0.2)
            .position([0.0, 0.0, 0.0])
            .velocity([0.3, 0.0, 0.0])
            .sleep_thresholds(0.5, 0.5),
    );
    let eager = world.spawn(
        BodyDesc::sphere(0.2)
            .position([10.0, 0.0, 0.0])
            .velocity([0.3, 0.0, 0.0]),
    );
    settle(&mut world, 40);
    assert!(
        world.read_state(lazy).sleeping,
        "overridden threshold must let the slow body sleep"
    );
    assert!(
        !world.read_state(eager).sleeping,
        "global threshold must keep the same body awake"
    );
}
