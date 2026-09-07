use super::common::{DT, distance, settle, sim, static_config};
use dynamis_model::{BodyDesc, ConstraintDesc, PhysicsConfig};
use dynamis_sim::Simulation;
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
