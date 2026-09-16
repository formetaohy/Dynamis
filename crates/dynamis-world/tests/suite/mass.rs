use super::common::{DT, flat_mesh_floor, observed_world, settle_until, static_config};
use dynamis_model::{BodyDesc, ColliderDesc, ConstraintDesc, Shape};

fn swing_angle(orientation: [f32; 4]) -> f32 {
    let w = orientation[3];
    let z = orientation[2];
    (2.0 * (z * w).atan2(w * w - z * z)).abs()
}

#[test]
fn offset_com_pendulum_swings_under_gravity() {
    let mut world = observed_world(super::common::gravity_config());
    let anchor = world.spawn(BodyDesc::static_sphere(0.1).position([0.0, 0.0, 0.0]));
    let bob = world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::sphere(0.2)).offset([1.5, 0.0, 0.0]))
            .position([0.0, 0.0, 0.0])
            .com([1.5, 0.0, 0.0])
            .mass(1.0),
    );
    world.add_constraint(
        anchor,
        bob,
        ConstraintDesc::revolute([0.0; 3], [0.0; 3], [0.0, 0.0, 1.0]).disable_collisions(true),
    );
    let mut max_angle = 0.0f32;
    settle_until(&mut world, 240, |world| {
        max_angle = max_angle.max(swing_angle(world.read_state(bob).orientation));
        max_angle > 0.2
    });
    assert!(
        max_angle > 0.2,
        "an offset center of mass must torque the pendulum, max {max_angle} rad"
    );
}

#[test]
fn composite_body_reports_center_of_mass() {
    let mut world = observed_world(static_config());
    let double = world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::sphere(0.2)))
            .collider(ColliderDesc::new(Shape::sphere(0.2)).offset([2.0, 0.0, 0.0]))
            .mass(2.0),
    );
    let state = world.read_state(double);
    assert_eq!(state.com, [1.0, 0.0, 0.0]);
    world.step(DT);
    world.wait();
    let state = world.read_state(double);
    assert_eq!(state.com, [1.0, 0.0, 0.0]);
}

#[test]
fn set_com_relocates_center_of_mass_on_gpu() {
    let mut world = observed_world(static_config());
    let double = world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::sphere(0.2)))
            .collider(ColliderDesc::new(Shape::sphere(0.2)).offset([2.0, 0.0, 0.0]))
            .mass(2.0),
    );
    world.set_com(double, [2.0, 0.0, 0.0]);
    world.step(DT);
    world.wait();
    let state = world.read_state(double);
    assert_eq!(state.com, [2.0, 0.0, 0.0]);
}

#[test]
fn custom_inertia_scales_angular_response() {
    let mut world = observed_world(static_config());
    let default = world.spawn(BodyDesc::sphere(0.5).position([-3.0, 0.0, 0.0]));
    let custom = world.spawn(BodyDesc::sphere(0.5).position([3.0, 0.0, 0.0]));
    world.set_inertia(custom, [0.05, 0.0, 0.0, 0.05, 0.0, 0.05]);
    let impulse = [0.0, 50.0, 0.0];
    world.apply_impulse_at_point(default, impulse, [-2.5, 0.0, 0.0]);
    world.apply_impulse_at_point(custom, impulse, [3.5, 0.0, 0.0]);
    world.step(DT);
    world.wait();
    let default_spin = world.read_state(default).angular_velocity[2];
    let custom_spin = world.read_state(custom).angular_velocity[2];
    assert!(
        custom_spin > default_spin * 1.5,
        "a lower inertia tensor must amplify the angular response, default {default_spin}, custom {custom_spin}"
    );
}

#[test]
fn ccd_stops_against_mesh_floor_while_non_ccd_passes() {
    let mut world = observed_world(static_config());
    flat_mesh_floor(&mut world);
    let shielded = world.spawn(
        BodyDesc::sphere(0.3)
            .position([0.0, 2.0, 0.0])
            .velocity([0.0, -60.0, 0.0])
            .ccd(true),
    );
    let unshielded = world.spawn(
        BodyDesc::sphere(0.3)
            .position([5.0, 2.0, 0.0])
            .velocity([0.0, -60.0, 0.0]),
    );
    for _ in 0..5 {
        world.step(DT);
    }
    world.wait();
    let shielded_y = world.read_state(shielded).position[1];
    let unshielded_y = world.read_state(unshielded).position[1];
    assert!(
        shielded_y > 0.2,
        "ccd must stop the ball above the mesh floor, got y {shielded_y}"
    );
    assert!(
        unshielded_y < -0.5,
        "without ccd the ball must tunnel through the mesh floor, got y {unshielded_y}"
    );
}

#[test]
fn ccd_retreats_before_mesh_impact_without_tunneling() {
    let mut world = observed_world(dynamis_model::PhysicsConfig {
        max_velocity: 1000.0,
        ..static_config()
    });
    flat_mesh_floor(&mut world);
    let ball = world.spawn(
        BodyDesc::sphere(0.3)
            .position([0.0, 5.0, 0.0])
            .velocity([0.0, -400.0, 0.0])
            .ccd(true),
    );
    world.step(DT);
    world.wait();
    let y = world.read_state(ball).position[1];
    assert!(
        y > 0.2 && y < 0.6,
        "a fast ball must stop at the mesh surface, got y {y}"
    );
}

fn octahedron(half: f32) -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let vertices = vec![
        [half, 0.0, 0.0],
        [-half, 0.0, 0.0],
        [0.0, half, 0.0],
        [0.0, -half, 0.0],
        [0.0, 0.0, half],
        [0.0, 0.0, -half],
    ];
    let triangles = vec![
        [0, 2, 4],
        [0, 4, 3],
        [0, 3, 5],
        [0, 5, 2],
        [1, 4, 2],
        [1, 3, 4],
        [1, 5, 3],
        [1, 2, 5],
    ];
    (vertices, triangles)
}

fn unit_tetrahedron() -> (Vec<[f32; 3]>, Vec<[u32; 3]>) {
    let vertices = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ];
    let triangles = vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
    (vertices, triangles)
}

#[test]
fn hull_density_uses_the_exact_hull_volume() {
    let mut world = observed_world(static_config());
    let (vertices, triangles) = octahedron(1.0);
    let hull = world.add_hull(&vertices, &triangles);
    let body = world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::hull(hull)))
            .density(1.5)
            .position([0.0, 5.0, 0.0]),
    );
    let state = world.read_state(body);
    let expected = 1.0 / (1.5 * 4.0 / 3.0);
    assert!(
        (state.inverse_mass - expected).abs() < 1e-6,
        "an octahedron hull must weigh its exact volume, got {}",
        state.inverse_mass
    );
    for axis in 0..3 {
        assert!(
            state.com[axis].abs() < 1e-6,
            "a symmetric hull must center at the origin, axis {axis} is {}",
            state.com[axis]
        );
    }
}

#[test]
fn an_asymmetric_hull_centers_on_its_geometry() {
    let mut world = observed_world(static_config());
    let (vertices, triangles) = unit_tetrahedron();
    let hull = world.add_hull(&vertices, &triangles);
    let body = world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::hull(hull)))
            .density(6.0)
            .position([0.0, 5.0, 0.0]),
    );
    let state = world.read_state(body);
    assert!(
        (state.inverse_mass - 1.0).abs() < 1e-6,
        "a unit tetrahedron at density 6 must weigh one, got {}",
        state.inverse_mass
    );
    for axis in 0..3 {
        assert!(
            (state.com[axis] - 0.25).abs() < 1e-6,
            "a tetrahedron hull must center at its geometry, axis {axis} is {}",
            state.com[axis]
        );
    }
}

#[test]
fn a_hull_spins_with_its_exact_inertia() {
    let mut world = observed_world(static_config());
    let (vertices, triangles) = octahedron(1.0);
    let hull = world.add_hull(&vertices, &triangles);
    let body = world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::hull(hull)))
            .density(1.0)
            .position([0.0, 5.0, 0.0]),
    );
    world.apply_angular_impulse(body, [0.0, 0.3, 0.0]);
    world.step(DT);
    world.wait();
    let state = world.read_state(body);
    let expected = 0.3 * 15.0 / 4.0;
    assert!(
        (state.angular_velocity[1] - expected).abs() < 1e-4,
        "an octahedron hull must answer an angular impulse with its exact inertia, got {}",
        state.angular_velocity[1]
    );
    assert!(state.angular_velocity[0].abs() < 1e-6);
    assert!(state.angular_velocity[2].abs() < 1e-6);
}

#[test]
fn a_hull_compound_centers_between_its_placed_geometries() {
    let mut world = observed_world(static_config());
    let (vertices, triangles) = unit_tetrahedron();
    let hull = world.add_hull(&vertices, &triangles);
    let body = world.spawn(
        BodyDesc::new(ColliderDesc::new(Shape::hull(hull)))
            .collider(ColliderDesc::new(Shape::hull(hull)).offset([3.0, 3.0, 3.0]))
            .density(6.0)
            .position([0.0, 5.0, 0.0]),
    );
    let state = world.read_state(body);
    assert!(
        (state.inverse_mass - 0.5).abs() < 1e-6,
        "two unit tetrahedra at density six must weigh one, got {}",
        state.inverse_mass
    );
    for axis in 0..3 {
        assert!(
            (state.com[axis] - 1.75).abs() < 1e-6,
            "a placed hull compound must center between its geometries, axis {axis} is {}",
            state.com[axis]
        );
    }
}
