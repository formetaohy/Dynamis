use super::common::{DT, distance, settle, sim, static_config};
use dynamis_model::{BodyDesc, ConstraintDesc, PhysicsConfig};
use dynamis_simulate::Simulation;
use std::panic::{AssertUnwindSafe, catch_unwind};

fn hinge_angle(orientation: [f32; 4]) -> f32 {
    let w = orientation[3];
    let z = orientation[2];
    (2.0 * (z * w).atan2(w * w - z * z)).abs()
}

#[test]
fn ball_constraint_keeps_bodies_linked() {
    let mut world = sim(8, static_config());
    let first = world.spawn(BodyDesc::sphere(0.2));
    let second = world.spawn(BodyDesc::sphere(0.2).position([1.0, 0.0, 0.0]));
    world.add_constraint(first, second, ConstraintDesc::ball([0.0; 3], [0.0; 3]));
    for _ in 0..60 {
        world.apply_force(first, [0.0, 5.0, 0.0]);
        world.step(DT);
    }
    world.wait();
    let span = distance(
        world.read_state(first).position,
        world.read_state(second).position,
    );
    assert!(
        span < 1.25,
        "ball constraint must keep bodies near their initial distance, got {span}"
    );
}

#[test]
fn distance_constraint_holds_span() {
    let mut world = sim(8, static_config());
    let first = world.spawn(BodyDesc::sphere(0.2));
    let second = world.spawn(BodyDesc::sphere(0.2).position([0.0, 3.0, 0.0]));
    world.add_constraint(
        first,
        second,
        ConstraintDesc::distance([0.0; 3], [0.0; 3], 3.0),
    );
    for _ in 0..60 {
        world.apply_force(first, [2.0, 0.0, 0.0]);
        world.step(DT);
    }
    world.wait();
    let span = distance(
        world.read_state(first).position,
        world.read_state(second).position,
    );
    assert!(
        (span - 3.0).abs() < 0.4,
        "distance constraint must hold the span, got {span}"
    );
}

#[test]
fn fixed_constraint_preserves_offset() {
    let mut world = sim(8, static_config());
    let first = world.spawn(BodyDesc::sphere(0.5));
    let second = world.spawn(BodyDesc::sphere(0.5).position([0.0, 0.0, 2.0]));
    world.add_constraint(first, second, ConstraintDesc::fixed([0.0; 3], [0.0; 3]));
    world.apply_force(first, [0.0, 0.0, 4.0]);
    settle(&mut world, 60);
    let separation = world.read_state(second).position[2] - world.read_state(first).position[2];
    assert!(
        (separation - 2.0).abs() < 0.3,
        "fixed constraint must preserve the offset, got {separation}"
    );
}

fn pendulum_world(
    limit: Option<(f32, f32)>,
    motor: Option<f32>,
) -> (Simulation, dynamis_model::BodyHandle) {
    let mut world = sim(
        8,
        PhysicsConfig {
            sleep_velocity: 0.0,
            sleep_angular_velocity: 0.0,
            ..static_config()
        },
    );
    let anchor = world.spawn(BodyDesc::static_sphere(0.1));
    let arm = world.spawn(BodyDesc::cuboid([0.1, 1.0, 0.1]).position([0.0, 1.0, 0.0]));
    let mut desc = ConstraintDesc::revolute([0.0; 3], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]);
    if let Some((min, max)) = limit {
        desc = desc.limit(min, max);
    }
    if let Some(speed) = motor {
        desc = desc.motor(speed);
    }
    world.add_constraint(anchor, arm, desc);
    (world, arm)
}

#[test]
fn revolute_rotates_freely_without_limit() {
    let (mut world, arm) = pendulum_world(None, None);
    world.apply_force(arm, [1.0, 0.0, 0.0]);
    let mut max_angle = 0.0f32;
    for _ in 0..120 {
        world.step(DT);
        world.wait();
        max_angle = max_angle.max(hinge_angle(world.read_state(arm).orientation));
    }
    assert!(
        max_angle > 0.01,
        "unlimited revolute must swing under a kick, max {max_angle} rad"
    );
}

#[test]
fn revolute_limit_caps_swing() {
    let (mut world, arm) = pendulum_world(Some((-0.2, 0.2)), None);
    for _ in 0..120 {
        world.apply_force(arm, [1.0, 0.0, 0.0]);
        world.step(DT);
    }
    world.wait();
    let angle = hinge_angle(world.read_state(arm).orientation);
    assert!(
        angle < 0.25,
        "revolute limit must cap the swing, got {angle} rad"
    );
}

#[test]
fn revolute_motor_drives_arm() {
    let (mut world, arm) = pendulum_world(None, Some(2.0));
    for _ in 0..90 {
        world.step(DT);
    }
    world.wait();
    let angle = hinge_angle(world.read_state(arm).orientation);
    assert!(
        angle > 1.0,
        "revolute motor must drive the arm, got {angle} rad"
    );
}

#[test]
fn prismatic_locks_perpendicular_motion() {
    let mut world = sim(8, static_config());
    let guide = world.spawn(BodyDesc::static_sphere(0.1).position([0.0, 5.0, 0.0]));
    let slider = world.spawn(BodyDesc::sphere(0.3).position([0.0, 6.0, 0.0]));
    world.add_constraint(
        guide,
        slider,
        ConstraintDesc::prismatic([0.0; 3], [0.0, -1.0, 0.0], [0.0, 0.0, 1.0]),
    );
    world.apply_force(slider, [3.0, 0.0, 0.0]);
    settle(&mut world, 120);
    let position = world.read_state(slider).position;
    assert!(
        (position[1] - 6.0).abs() < 0.15,
        "prismatic must hold the axis offset, got {:?}",
        position
    );
    assert!(
        position[0].abs() < 0.2,
        "prismatic must resist cross-axis force, got {:?}",
        position
    );
}

#[test]
fn prismatic_motor_drives_and_limit_caps_travel() {
    let mut world = sim(8, static_config());
    let anchor = world.spawn(BodyDesc::static_sphere(0.1));
    let slider = world.spawn(BodyDesc::cuboid([0.1, 0.1, 0.1]).position([1.0, 0.0, 0.0]));
    world.add_constraint(
        anchor,
        slider,
        ConstraintDesc::prismatic([0.0; 3], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0])
            .limit(0.0, 2.0)
            .motor(2.0),
    );
    for _ in 0..90 {
        world.step(DT);
    }
    world.wait();
    let x = world.read_state(slider).position[0];
    assert!(
        (x - 1.5).abs() < 0.05,
        "prismatic limit must cap the travel at 1.5, got x={x}"
    );
}

#[test]
fn prismatic_motor_alone_drives_beyond_start() {
    let mut world = sim(8, static_config());
    let anchor = world.spawn(BodyDesc::static_sphere(0.1));
    let slider = world.spawn(BodyDesc::cuboid([0.1, 0.1, 0.1]).position([1.0, 0.0, 0.0]));
    world.add_constraint(
        anchor,
        slider,
        ConstraintDesc::prismatic([0.0; 3], [0.5, 0.0, 0.0], [1.0, 0.0, 0.0]).motor(2.0),
    );
    for _ in 0..60 {
        world.step(DT);
    }
    world.wait();
    let x = world.read_state(slider).position[0];
    assert!(
        x > 2.5,
        "motor must drive the slider along its axis, got x={x}"
    );
}

#[test]
fn distance_spring_sags_under_gravity() {
    let mut world = sim(
        8,
        PhysicsConfig {
            sleep_velocity: 0.0,
            sleep_angular_velocity: 0.0,
            ..PhysicsConfig::default()
        },
    );
    let anchor = world.spawn(BodyDesc::static_sphere(0.1));
    let ball = world.spawn(BodyDesc::sphere(0.2).position([0.0, 1.0, 0.0]));
    world.add_constraint(
        anchor,
        ball,
        ConstraintDesc::distance([0.0; 3], [0.0; 3], 1.0).spring(1.0, 0.5),
    );
    settle(&mut world, 240);
    let y = world.read_state(ball).position[1];
    assert!(
        y > 0.3 && y < 1.0,
        "spring must sag below the rest length, got y={y}"
    );
}

#[test]
fn joined_bodies_collision_policy_controls_overlap() {
    let mut merged = sim(8, static_config());
    let first = merged.spawn(BodyDesc::sphere(0.5));
    let second = merged.spawn(BodyDesc::sphere(0.5).position([0.0, 0.6, 0.0]));
    merged.add_constraint(first, second, ConstraintDesc::ball([0.0; 3], [0.0; 3]));
    settle(&mut merged, 5);
    let y = merged.read_state(second).position[1];
    assert!(
        y < 0.75,
        "disabled collisions must not push joined bodies apart, got y={y}"
    );

    let mut separated = sim(8, static_config());
    let first = separated.spawn(BodyDesc::sphere(0.5));
    let second = separated.spawn(BodyDesc::sphere(0.5).position([0.0, 0.6, 0.0]));
    separated.add_constraint(
        first,
        second,
        ConstraintDesc::ball([0.0; 3], [0.0; 3]).disable_collisions(false),
    );
    settle(&mut separated, 30);
    let y = separated.read_state(second).position[1];
    assert!(
        y > 0.75,
        "enabled collisions must push joined bodies apart, got y={y}"
    );
}

#[test]
fn constraint_handle_reuse_bumps_generation() {
    let mut world = sim(16, static_config());
    let first = world.spawn(BodyDesc::sphere(0.2));
    let second = world.spawn(BodyDesc::sphere(0.2).position([1.0, 0.0, 0.0]));
    let constraint = world.add_constraint(first, second, ConstraintDesc::ball([0.0; 3], [0.0; 3]));
    world.remove_constraint(constraint);
    let fresh = world.add_constraint(first, second, ConstraintDesc::ball([0.0; 3], [0.0; 3]));
    assert_ne!(constraint.generation, fresh.generation);
    assert_eq!(world.constraints(), &[fresh]);
    world.remove_constraint(fresh);
    world.remove(first);
    assert_eq!(world.count(), 1);
}

#[test]
fn constraint_misuse_panics() {
    let mut world = sim(4, static_config());
    let first = world.spawn(BodyDesc::sphere(0.2));
    let second = world.spawn(BodyDesc::sphere(0.2).position([1.0, 0.0, 0.0]));
    let third = world.spawn(BodyDesc::sphere(0.2).position([0.0, 1.0, 0.0]));
    world.add_constraint(first, second, ConstraintDesc::ball([0.0; 3], [0.0; 3]));
    assert!(
        catch_unwind(AssertUnwindSafe(|| world.remove(first))).is_err(),
        "removing a constrained body must panic"
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            world.add_constraint(first, second, ConstraintDesc::ball([0.0; 3], [0.0; 3]));
            world.add_constraint(second, first, ConstraintDesc::ball([0.0; 3], [0.0; 3]));
            world.add_constraint(first, third, ConstraintDesc::ball([0.0; 3], [0.0; 3]));
            world.add_constraint(second, third, ConstraintDesc::ball([0.0; 3], [0.0; 3]));
        }))
        .is_err(),
        "exceeding constraint capacity must panic"
    );
}

#[test]
fn removing_constraint_allows_body_removal() {
    let mut world = sim(8, static_config());
    let first = world.spawn(BodyDesc::sphere(0.2));
    let second = world.spawn(BodyDesc::sphere(0.2).position([1.0, 0.0, 0.0]));
    let constraint = world.add_constraint(first, second, ConstraintDesc::ball([0.0; 3], [0.0; 3]));
    world.remove_constraint(constraint);
    world.remove(first);
    assert_eq!(world.count(), 1);
    assert!(world.constraints().is_empty());
}

fn twist_angle(orientation: [f32; 4]) -> f32 {
    let q = orientation;
    let y = q[1];
    let w = q[3];
    (2.0 * (y * w).atan2(w * w - y * y)).abs()
}

#[test]
fn ball_twist_limit_caps_relative_rotation() {
    let mut world = sim(8, static_config());
    let first = world.spawn(BodyDesc::sphere(0.2));
    let second = world.spawn(BodyDesc::sphere(0.2).position([1.0, 0.0, 0.0]));
    world.add_constraint(
        first,
        second,
        ConstraintDesc::ball([0.0; 3], [0.0; 3])
            .axis([0.0, 0.0, 1.0])
            .limit(-0.3, 0.3),
    );
    for _ in 0..120 {
        world.apply_torque(second, [0.0, 0.0, 12.0]);
        world.step(DT);
    }
    world.wait();
    let state = world.read_state(second);
    let angle = twist_angle(state.orientation);
    assert!(
        angle < 0.5,
        "twist limit must cap relative rotation, got {angle} rad"
    );
}

#[test]
fn ball_swing_limit_caps_conical_sway() {
    let mut world = sim(8, static_config());
    let first = world.spawn(BodyDesc::sphere(0.2));
    let second = world.spawn(BodyDesc::sphere(0.2).position([1.0, 0.0, 0.0]));
    world.add_constraint(
        first,
        second,
        ConstraintDesc::ball([0.0; 3], [0.0; 3]).swing(0.2, 0.2),
    );
    for _ in 0..120 {
        world.apply_torque(second, [0.0, 20.0, 0.0]);
        world.step(DT);
    }
    world.wait();
    let state = world.read_state(second);
    let q = state.orientation;
    let axis = 2.0 * (q[0] * q[0] + q[2] * q[2]).sqrt();
    let tilt = 2.0 * axis.clamp(0.0, 1.0).asin();
    println!("swing q={q:?} tilt={tilt}");
    assert!(
        tilt < 0.45,
        "swing limit must cap the sway angle, got {tilt} rad"
    );
}

#[test]
fn gear_constraint_links_angular_velocities() {
    let mut world = sim(8, static_config());
    let _anchor = world.spawn(BodyDesc::static_sphere(0.1));
    let first = world.spawn(BodyDesc::sphere(0.3).position([0.0, 0.0, 0.0]));
    let second = world.spawn(BodyDesc::sphere(0.3).position([1.0, 0.0, 0.0]));
    world.add_constraint(
        first,
        second,
        ConstraintDesc::gear([0.0, 0.0, 1.0], [0.0, 0.0, 1.0], 2.0),
    );
    world.set_angular_velocity(first, [0.0, 0.0, 10.0]);
    for _ in 0..60 {
        world.step(DT);
    }
    world.wait();
    let a = world.read_state(first).angular_velocity[2];
    let b = world.read_state(second).angular_velocity[2];
    assert!(
        (b - 2.0 * a).abs() < 1.0,
        "gear must enforce omega_b = 2 * omega_a, got a={a} b={b}"
    );
}

#[test]
fn pulley_constraint_holds_rope_length() {
    let mut world = sim(8, static_config());
    let _anchor = world.spawn(BodyDesc::static_sphere(0.1));
    let first = world.spawn(BodyDesc::sphere(0.2).position([0.0, 1.0, 0.0]));
    let second = world.spawn(BodyDesc::sphere(0.2).position([2.0, 1.0, 0.0]));
    let fixed_a = [0.0f32, 3.0, 0.0];
    let fixed_b = [2.0f32, 3.0, 0.0];
    let length = 2.0 + 2.0;
    world.add_constraint(
        first,
        second,
        ConstraintDesc::pulley([0.0; 3], [0.0; 3], fixed_a, fixed_b, length),
    );
    for _ in 0..90 {
        world.apply_force(first, [0.0, 6.0, 0.0]);
        world.step(DT);
    }
    world.wait();
    let pa = world.read_state(first).position;
    let pb = world.read_state(second).position;
    let rope = distance(pa, fixed_a) + distance(pb, fixed_b);
    assert!(
        (rope - length).abs() < 0.3,
        "pulley must hold the rope length, got {rope} vs {length}"
    );
}

#[test]
fn break_threshold_removes_constraint_under_load() {
    let mut world = sim(8, static_config());
    let first = world.spawn(BodyDesc::sphere(0.3));
    let second = world.spawn(BodyDesc::sphere(0.3).position([1.0, 0.0, 0.0]));
    let handle = world.add_constraint(
        first,
        second,
        ConstraintDesc::ball([0.0; 3], [0.3, 0.0, 0.0]).break_threshold(0.4, 0.0),
    );
    for _ in 0..90 {
        world.apply_force(second, [0.0, 40.0, 0.0]);
        world.step(DT);
    }
    world.wait();
    assert!(
        !world.constraints().contains(&handle),
        "overloaded ball constraint must break and be removed"
    );
    let state = world.read_state(second);
    assert!(
        state.position[1] > 1.1,
        "broken body must be free to move, got {:?}",
        state.position
    );
}

#[test]
fn motor_force_cap_limits_driving_torque() {
    let mut world = sim(8, static_config());
    let first = world.spawn(BodyDesc::sphere(0.3));
    let second = world.spawn(BodyDesc::sphere(0.3).position([1.0, 0.0, 0.0]));
    world.add_constraint(
        first,
        second,
        ConstraintDesc::revolute([0.0; 3], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0])
            .motor(20.0)
            .motor_force(0.5),
    );
    for _ in 0..120 {
        world.step(DT);
    }
    world.wait();
    let spin = world.read_state(second).angular_velocity[2].abs();
    assert!(
        spin < 6.0,
        "a tiny motor force must cap the driven spin, got {spin}"
    );
}
